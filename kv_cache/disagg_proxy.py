# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import zmq

def main():
    port = 5555
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
        
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{port}")
    print(f"Disagg Discovery Proxy started on port {port}", flush=True)
    
    store = {}
    
    while True:
        try:
            msg = socket.recv_string()
            parts = msg.split(":")
            cmd = parts[0]
            
            if cmd == "REGISTER":
                # REGISTER:<name>:<ip>:<zmq_port>:<transport_port>
                if len(parts) != 5:
                    socket.send_string("ERROR:Invalid REGISTER format")
                    continue
                name, ip, zmq_port, trans_port = parts[1], parts[2], parts[3], parts[4]
                store[name] = (ip, zmq_port, trans_port)
                print(f"Registered {name} at {ip} (ZMQ:{zmq_port}, Trans:{trans_port})", flush=True)
                socket.send_string("OK")
                
            elif cmd == "RESOLVE":
                # RESOLVE:<name>
                if len(parts) != 2:
                    socket.send_string("ERROR:Invalid RESOLVE format")
                    continue
                name = parts[1]
                if name in store:
                    ip, zmq_port, trans_port = store[name]
                    socket.send_string(f"OK:{ip}:{zmq_port}:{trans_port}")
                else:
                    socket.send_string("ERROR:NotFound")
            else:
                socket.send_string("ERROR:UnknownCommand")
        except Exception as e:
            print(f"Error: {e}", flush=True)
            try:
                socket.send_string(f"ERROR:{e}")
            except:
                pass

if __name__ == "__main__":
    main()
