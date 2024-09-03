import sys
import os
import signal
import time
import logging
import numpy as np
import socket
import struct

try:
    import cv2 as cv
except ImportError:
    print("Please install OpenCV to see the thermal image")
    sys.exit(1)

from senxor.mi48 import MI48, format_header, format_framestats
from senxor.utils import data_to_frame, remap, cv_filter, cv_render, RollingAverageFilter, connect_senxor

# Logging setup
logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))
logger = logging.getLogger(__name__)

# Global variable for the MI48 instance
global mi48

# Signal handler to ensure clean closure
def signal_handler(sig, frame):
    """Ensure clean exit in case of SIGINT or SIGTERM"""
    logger.info("Exiting due to SIGINT or SIGTERM")
    mi48.stop()
    cv.destroyAllWindows()
    logger.info("Done.")
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Connect to the thermal camera
mi48, connected_port, port_names = connect_senxor()
logger.info(f"Connected to camera on port: {connected_port}")
logger.info('Camera info:')
logger.info(mi48.camera_info)

# Set desired FPS
STREAM_FPS = int(sys.argv[1]) if len(sys.argv) == 2 else 15
mi48.set_fps(STREAM_FPS)

# Setup camera filters and start acquisition
mi48.disable_filter(f1=True, f2=True, f3=True)
mi48.set_filter_1(85)
mi48.enable_filter(f1=True, f2=False, f3=False, f3_ks_5=False)
mi48.set_offset_corr(0.0)
mi48.set_sens_factor(100)
mi48.get_sens_factor()
mi48.start(stream=True, with_header=True)

# GUI setup
GUI = False

# Filter parameters and rolling average filters for temperature bounds
par = {'blur_ks': 3, 'd': 5, 'sigmaColor': 27, 'sigmaSpace': 27}
dminav = RollingAverageFilter(N=10)
dmaxav = RollingAverageFilter(N=10)

# Server setup
HOST = '0.0.0.0'  # Listen on all interfaces
PORT = 65432      # Port to listen on

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen()

logger.info(f"Server listening on {HOST}:{PORT}")

def format_data(data):
    """Format the thermal data according to the binary protocol."""
    start_package = 0xaabbccdd
    num_pixels = 0x1360
    num_rows = 0x0050
    temp_data = (data * 10).astype(np.uint16).flatten()
    data_sum = np.sum(temp_data, dtype=np.uint32)
    end_package = 0xddccbbaa

    package = struct.pack(
        ">IHH4960H",  # Format string for the struct (4960 = 80x62 pixels)
        start_package,
        num_pixels,
        num_rows,
        *temp_data
    )
    package += struct.pack(">I", data_sum)
    package += struct.pack(">I", end_package)

    return package

def handle_client(client_socket):
    """Handle the client connection."""
    try:
        while True:
            data, header = mi48.read()
            if data is None:
                logger.warning("No data received from camera.")
                break

            min_temp = dminav(data.min())
            max_temp = dmaxav(data.max())
            frame = data_to_frame(data, (80, 62), hflip=False)
            frame = np.clip(frame, min_temp, max_temp)

            filt_uint8 = cv_filter(remap(frame), par, use_median=True,
                                   use_bilat=True, use_nlm=False)

            # Format the data according to the protocol
            packet = format_data(frame)

            # Send the data to the client
            client_socket.sendall(packet)

            if header is not None:
                logger.debug('  '.join([format_header(header),
                                        format_framestats(data)]))
            else:
                logger.debug(format_framestats(data))

            if GUI:
                cv_render(filt_uint8, resize=(400, 310), colormap='rainbow2')
                key = cv.waitKey(1)
                if key == ord("q"):
                    break
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        client_socket.close()
        logger.info("Client connection closed.")

# Main server loop
try:
    while True:
        logger.info("Waiting for a client to connect...")
        client_socket, client_address = server_socket.accept()
        logger.info(f"Connected to client: {client_address}")
        handle_client(client_socket)
except KeyboardInterrupt:
    logger.info("Server shutting down.")
finally:
    mi48.stop()
    cv.destroyAllWindows()
    server_socket.close()
    logger.info("Server stopped.")
