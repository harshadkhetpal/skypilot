"""SSH WebSocket proxy handlers for SkyPilot API Server.

This module contains the SSH proxy handlers extracted from server.py,
enabling them to be reused by other FastAPI applications (e.g., the
SkyPilot Agent).
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from enum import IntEnum
import os
import shlex
import socket
import struct
import typing
from typing import Awaitable, Callable, Optional, Type

import fastapi

from sky import clouds
from sky import core
from sky import sky_logging
from sky.metrics import utils as metrics_utils
from sky.provision.slurm import utils as slurm_utils
from sky.utils import command_runner
from sky.utils import context_utils
from sky.utils import env_options
from sky.utils import interactive_utils
from sky.utils import status_lib

if typing.TYPE_CHECKING:
    from sky import backends

logger = sky_logging.init_logger(__name__)

ssh_router = fastapi.APIRouter(tags=['ssh-proxy'])


class SSHMessageType(IntEnum):
    REGULAR_DATA = 0
    PINGPONG = 1
    LATENCY_MEASUREMENT = 2


async def _get_cluster_and_validate(
    cluster_name: str,
    cloud_type: Type[clouds.Cloud],
) -> 'backends.CloudVmRayResourceHandle':
    """Fetch cluster status and validate it's UP and correct cloud type."""
    # Run core.status in another thread to avoid blocking the event loop.
    # Use summary_response=True to skip expensive DB columns (owner, metadata,
    # last_creation_yaml) and cluster event queries that are unnecessary for
    # simple cluster validation. This keeps per-call overhead low enough to
    # handle 20+ concurrent WebSocket SSH connections without timeout.
    # TODO(aylei): core.status() will be called with server user, which has
    # permission to all workspaces, this will break workspace isolation.
    # It is ok for now, as users with limited access will not get the ssh config
    # for the clusters in non-accessible workspaces.
    with ThreadPoolExecutor(max_workers=1) as thread_pool_executor:
        cluster_records = await context_utils.to_thread_with_executor(
            thread_pool_executor,
            core.status,
            cluster_name,
            all_users=True,
            summary_response=True)

    if not cluster_records:
        raise fastapi.HTTPException(status_code=404,
                                    detail=f'Cluster {cluster_name} not found')
    cluster_record = cluster_records[0]

    if cluster_record['status'] not in (status_lib.ClusterStatus.INIT,
                                        status_lib.ClusterStatus.UP,
                                        status_lib.ClusterStatus.AUTOSTOPPING):
        raise fastapi.HTTPException(
            status_code=400, detail=f'Cluster {cluster_name} is not running')

    handle: Optional['backends.CloudVmRayResourceHandle'] = cluster_record[
        'handle']
    assert handle is not None, 'Cluster handle is None'
    if not isinstance(handle.launched_resources.cloud, cloud_type):
        raise fastapi.HTTPException(
            status_code=400,
            detail=f'Cluster {cluster_name} is not a {str(cloud_type())} '
            'cluster. Use ssh to connect to the cluster instead.')

    return handle


async def _run_websocket_proxy(
    websocket: fastapi.WebSocket,
    read_from_backend: Callable[[], Awaitable[bytes]],
    write_to_backend: Callable[[bytes], Awaitable[None]],
    close_backend: Callable[[], Awaitable[None]],
    timestamps_supported: bool,
) -> bool:
    """Run bidirectional WebSocket-to-backend proxy.

    Args:
        websocket: FastAPI WebSocket connection
        read_from_backend: Async callable to read bytes from backend
        write_to_backend: Async callable to write bytes to backend
        close_backend: Async callable to close backend connection
        timestamps_supported: Whether to use message type framing

    Returns:
        True if SSH failed, False otherwise
    """
    ssh_failed = False
    websocket_closed = False

    async def websocket_to_backend():
        try:
            async for message in websocket.iter_bytes():
                if timestamps_supported:
                    type_size = struct.calcsize('!B')
                    message_type = struct.unpack('!B', message[:type_size])[0]
                    if message_type == SSHMessageType.REGULAR_DATA:
                        # Regular data - strip type byte and forward to backend
                        message = message[type_size:]
                    elif message_type == SSHMessageType.PINGPONG:
                        # PING message - respond with PONG
                        ping_id_size = struct.calcsize('!I')
                        if len(message) != type_size + ping_id_size:
                            raise ValueError(
                                f'Invalid PING message length: {len(message)}')
                        # Return the same PING message for latency measurement
                        await websocket.send_bytes(message)
                        continue
                    elif message_type == SSHMessageType.LATENCY_MEASUREMENT:
                        # Latency measurement from client
                        latency_size = struct.calcsize('!Q')
                        if len(message) != type_size + latency_size:
                            raise ValueError('Invalid latency measurement '
                                             f'message length: {len(message)}')
                        avg_latency_ms = struct.unpack(
                            '!Q',
                            message[type_size:type_size + latency_size])[0]
                        latency_seconds = avg_latency_ms / 1000
                        metrics_utils.SKY_APISERVER_WEBSOCKET_SSH_LATENCY_SECONDS.labels(  # pylint: disable=line-too-long
                            pid=os.getpid()).observe(latency_seconds)
                        continue
                    else:
                        raise ValueError(
                            f'Unknown message type: {message_type}')

                try:
                    await write_to_backend(message)
                except Exception as e:  # pylint: disable=broad-except
                    # Typically we will not reach here, if the conn to backend
                    # is disconnected, backend_to_websocket will exit first.
                    # But just in case.
                    logger.error(f'Failed to write to backend through '
                                 f'connection: {e}')
                    nonlocal ssh_failed
                    ssh_failed = True
                    break
        except fastapi.WebSocketDisconnect:
            pass
        nonlocal websocket_closed
        websocket_closed = True
        await close_backend()

    async def backend_to_websocket():
        try:
            while True:
                data = await read_from_backend()
                if not data:
                    if not websocket_closed:
                        logger.warning(
                            'SSH connection to backend is disconnected '
                            'before websocket connection is closed')
                        nonlocal ssh_failed
                        ssh_failed = True
                    break
                if timestamps_supported:
                    # Prepend message type byte (0 = regular data)
                    message_type_bytes = struct.pack(
                        '!B', SSHMessageType.REGULAR_DATA.value)
                    data = message_type_bytes + data
                await websocket.send_bytes(data)
        except Exception:  # pylint: disable=broad-except
            pass
        try:
            await websocket.close()
        except Exception:  # pylint: disable=broad-except
            # The websocket might have been closed by the client
            pass

    await asyncio.gather(websocket_to_backend(),
                         backend_to_websocket(),
                         return_exceptions=True)

    return ssh_failed


@ssh_router.websocket('/kubernetes-pod-ssh-proxy')
async def kubernetes_pod_ssh_proxy(
        websocket: fastapi.WebSocket,
        cluster_name: str,
        client_version: Optional[int] = None) -> None:
    """Proxies SSH to the Kubernetes pod with websocket."""
    await websocket.accept()
    logger.info(f'WebSocket connection accepted for cluster: {cluster_name}')

    timestamps_supported = client_version is not None and client_version > 21
    logger.info(f'Websocket timestamps supported: {timestamps_supported}, \
        client_version = {client_version}')

    handle = await _get_cluster_and_validate(cluster_name, clouds.Kubernetes)

    kubectl_cmd = handle.get_command_runners()[0].port_forward_command(
        port_forward=[(None, 22)])
    proc = await asyncio.create_subprocess_exec(
        *kubectl_cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT)
    logger.info(f'Started kubectl port-forward with command: {kubectl_cmd}')

    # Wait for port-forward to be ready and get the local port
    local_port = None
    assert proc.stdout is not None
    while True:
        stdout_line = await proc.stdout.readline()
        if stdout_line:
            decoded_line = stdout_line.decode()
            logger.info(f'kubectl port-forward stdout: {decoded_line}')
            if 'Forwarding from 127.0.0.1' in decoded_line:
                port_str = decoded_line.split(':')[-1]
                local_port = int(port_str.replace(' -> ', ':').split(':')[0])
                break
        else:
            await websocket.close()
            return

    logger.info(f'Starting port-forward to local port: {local_port}')
    conn_gauge = metrics_utils.SKY_APISERVER_WEBSOCKET_CONNECTIONS.labels(
        pid=os.getpid())
    ssh_failed = False
    try:
        conn_gauge.inc()
        # Connect to the local port
        reader, writer = await asyncio.open_connection('127.0.0.1', local_port)

        async def write_and_drain(data: bytes) -> None:
            writer.write(data)
            await writer.drain()

        async def close_writer() -> None:
            writer.close()

        ssh_failed = await _run_websocket_proxy(
            websocket,
            read_from_backend=lambda: reader.read(1024),
            write_to_backend=write_and_drain,
            close_backend=close_writer,
            timestamps_supported=timestamps_supported,
        )
    finally:
        conn_gauge.dec()
        reason = ''
        try:
            logger.info('Terminating kubectl port-forward process')
            proc.terminate()
        except ProcessLookupError:
            stdout = await proc.stdout.read()
            logger.error('kubectl port-forward was terminated before the '
                         'ssh websocket connection was closed. Remaining '
                         f'output: {str(stdout)}')
            reason = 'KubectlPortForwardExit'
            metrics_utils.SKY_APISERVER_WEBSOCKET_CLOSED_TOTAL.labels(
                pid=os.getpid(), reason=reason).inc()
        else:
            if ssh_failed:
                reason = 'SSHToPodDisconnected'
            else:
                reason = 'ClientClosed'
        metrics_utils.SKY_APISERVER_WEBSOCKET_CLOSED_TOTAL.labels(
            pid=os.getpid(), reason=reason).inc()


@ssh_router.websocket('/slurm-job-ssh-proxy')
async def slurm_job_ssh_proxy(websocket: fastapi.WebSocket,
                              cluster_name: str,
                              worker: int = 0,
                              client_version: Optional[int] = None) -> None:
    """Proxies SSH to the Slurm job via sshd inside srun."""
    await websocket.accept()
    logger.info(f'WebSocket connection accepted for cluster: '
                f'{cluster_name}')

    timestamps_supported = client_version is not None and client_version > 21
    logger.info(f'Websocket timestamps supported: {timestamps_supported}, \
        client_version = {client_version}')

    handle = await _get_cluster_and_validate(cluster_name, clouds.Slurm)

    assert handle.cached_cluster_info is not None, 'Cached cluster info is None'
    provider_config = handle.cached_cluster_info.provider_config
    assert provider_config is not None, 'Provider config is None'
    login_node_ssh_config = provider_config['ssh']
    login_node_host = login_node_ssh_config['hostname']
    login_node_port = int(login_node_ssh_config['port'])
    login_node_user = login_node_ssh_config['user']
    login_node_key = login_node_ssh_config.get('private_key', None)
    login_node_proxy_command = login_node_ssh_config.get('proxycommand', None)
    login_node_proxy_jump = login_node_ssh_config.get('proxyjump', None)

    login_node_runner = command_runner.SSHCommandRunner(
        (login_node_host, login_node_port),
        login_node_user,
        login_node_key,
        ssh_proxy_command=login_node_proxy_command,
        ssh_proxy_jump=login_node_proxy_jump,
    )

    ssh_cmd = login_node_runner.ssh_base_command(
        ssh_mode=command_runner.SshMode.NON_INTERACTIVE,
        port_forward=None,
        connect_timeout=None)

    # There can only be one InstanceInfo per instance_id.
    head_instance = handle.cached_cluster_info.get_head_instance()
    assert head_instance is not None, 'Head instance is None'
    job_id = head_instance.tags['job_id']

    # Instances are ordered: head first, then workers
    instances = handle.cached_cluster_info.instances
    node_hostnames = [inst[0].tags['node'] for inst in instances.values()]
    if worker >= len(node_hostnames):
        raise fastapi.HTTPException(
            status_code=400,
            detail=f'Worker index {worker} out of range. '
            f'Cluster has {len(node_hostnames)} nodes.')
    target_node = node_hostnames[worker]

    # Run sshd inside the Slurm job "container" via srun, such that it inherits
    # the resource constraints of the Slurm job.
    is_container_image = handle.launched_resources.extract_docker_image(
    ) is not None
    ssh_cmd += [
        shlex.quote(
            slurm_utils.srun_sshd_command(
                job_id,
                target_node,
                login_node_user,
                handle.cluster_name_on_cloud,
                is_container_image,
            ))
    ]

    proc = await asyncio.create_subprocess_shell(
        ' '.join(ssh_cmd),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,  # Capture stderr separately for logging
    )
    assert proc.stdin is not None
    assert proc.stdout is not None
    assert proc.stderr is not None

    stdin = proc.stdin
    stdout = proc.stdout
    stderr = proc.stderr

    async def log_stderr():
        while True:
            line = await stderr.readline()
            if not line:
                break
            logger.debug(f'srun stderr: {line.decode().rstrip()}')

    stderr_task = None
    if env_options.Options.SHOW_DEBUG_INFO.get():
        stderr_task = asyncio.create_task(log_stderr())
    conn_gauge = metrics_utils.SKY_APISERVER_WEBSOCKET_CONNECTIONS.labels(
        pid=os.getpid())
    ssh_failed = False
    try:
        conn_gauge.inc()

        async def write_and_drain(data: bytes) -> None:
            stdin.write(data)
            await stdin.drain()

        async def close_stdin() -> None:
            stdin.close()

        ssh_failed = await _run_websocket_proxy(
            websocket,
            read_from_backend=lambda: stdout.read(4096),
            write_to_backend=write_and_drain,
            close_backend=close_stdin,
            timestamps_supported=timestamps_supported,
        )

    finally:
        conn_gauge.dec()
        reason = ''
        try:
            logger.info('Terminating srun process')
            proc.terminate()
        except ProcessLookupError:
            stdout_data = await stdout.read()
            logger.error('srun process was terminated before the '
                         'ssh websocket connection was closed. Remaining '
                         f'output: {str(stdout_data)}')
            reason = 'SrunProcessExit'
            metrics_utils.SKY_APISERVER_WEBSOCKET_CLOSED_TOTAL.labels(
                pid=os.getpid(), reason=reason).inc()
        else:
            if ssh_failed:
                reason = 'SSHToSlurmJobDisconnected'
            else:
                reason = 'ClientClosed'

        metrics_utils.SKY_APISERVER_WEBSOCKET_CLOSED_TOTAL.labels(
            pid=os.getpid(), reason=reason).inc()

        # Cancel the stderr logging task if it's still running
        if stderr_task is not None and not stderr_task.done():
            stderr_task.cancel()
            try:
                await stderr_task
            except asyncio.CancelledError:
                pass


@ssh_router.websocket('/ssh-interactive-auth')
async def ssh_interactive_auth(websocket: fastapi.WebSocket,
                               session_id: str) -> None:
    """Proxies PTY for SSH interactive authentication via websocket.

    This endpoint receives a PTY file descriptor from a worker process
    and bridges it bidirectionally with a websocket connection, allowing
    the client to handle interactive SSH authentication (e.g., 2FA).

    Detects auth completion by monitoring terminal echo state and data flow.
    """
    await websocket.accept()
    logger.info(f'WebSocket connection accepted for SSH auth session: '
                f'{session_id}')

    loop = asyncio.get_running_loop()

    # Connect to worker process to receive PTY file descriptor
    fd_socket_path = interactive_utils.get_pty_socket_path(session_id)
    fd_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    master_fd = -1
    try:
        # Connect to worker's FD-passing socket
        await loop.sock_connect(fd_sock, fd_socket_path)
        master_fd = await loop.run_in_executor(None, interactive_utils.recv_fd,
                                               fd_sock)
        logger.debug(f'Received PTY master fd {master_fd} for session '
                     f'{session_id}')

        # Bridge PTY ↔ websocket bidirectionally
        async def websocket_to_pty():
            """Forward websocket messages to PTY."""
            try:
                async for message in websocket.iter_bytes():
                    await loop.run_in_executor(None, os.write, master_fd,
                                               message)
            except fastapi.WebSocketDisconnect:
                logger.debug(f'WebSocket disconnected for session {session_id}')
            except asyncio.CancelledError:
                pass
            except Exception as e:  # pylint: disable=broad-except
                logger.error(f'Error in websocket_to_pty: {e}')

        async def pty_to_websocket():
            """Forward PTY output to websocket and detect auth completion.

            Detects auth completion by monitoring terminal echo state.
            Echo is disabled during password prompts and enabled after
            successful authentication. Auth is considered complete when
            echo has been enabled for a sustained period (1s).
            """
            try:
                while True:
                    try:
                        data = await loop.run_in_executor(
                            None, os.read, master_fd, 4096)
                    except OSError as e:
                        logger.error(f'PTY read error (likely closed): {e}')
                        break

                    if not data:
                        break

                    await websocket.send_bytes(data)
            except asyncio.CancelledError:
                pass
            except Exception as e:  # pylint: disable=broad-except
                logger.error(f'Error in pty_to_websocket: {e}')
            finally:
                try:
                    await websocket.close()
                except Exception:  # pylint: disable=broad-except
                    pass

        await asyncio.gather(websocket_to_pty(), pty_to_websocket())

    except Exception as e:  # pylint: disable=broad-except
        logger.error(f'Error in SSH interactive auth websocket: {e}')
        raise
    finally:
        # Clean up
        if master_fd >= 0:
            try:
                os.close(master_fd)
            except OSError:
                pass
        fd_sock.close()
        logger.debug(f'SSH interactive auth session {session_id} completed')
