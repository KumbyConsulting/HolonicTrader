# === WEB DASHBOARD V2 (WebSocket Edition) ===
# Removed Eventlet for Windows Compatibility (using 'threading' mode)

from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import multiprocessing
import os
import sys
import time
import queue
import logging
from datetime import datetime

# Import Bot Logic
# Import Bot Logic (Graceful Fallback)
try:
    from main_live_phase4 import run_bot
    dependency_error = None
except ImportError as e:
    print(f">> [WARNING] Bot dependencies missing: {e}")
    run_bot = None
    dependency_error = str(e)

# Configuration
HOST = '0.0.0.0'
PORT = 5000
DEBUG = False # Set to False for Production/Stability

# Flask App
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}}) # Enable CORS for API endpoints
app.config['SECRET_KEY'] = 'holonic_secret_key_change_me'
# Windows: Use 'threading' to avoid Eventlet/Multiprocessing conflicts
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global State
manager = None
status_queue = None
command_queue = None
stop_event = None
bot_process = None

# Shared State Memory
SYSTEM_STATE = {
    'status': 'STOPPED',
    'regime': 'UNKNOWN',
    'health': 100.0,
    'equity': 0.0,
    'balance': 0.0,
    'pnl': 0.0,
    'win_rate': '0%',
    'positions': [],
    'radar': [],
    'logs': [],
    'equity_history': [], # timestamp, value
    'last_update': 0
}

# --- Monitor Thread (The Bridge) ---
def status_monitor():
    """
    Consumes messages from the Bot (Multiprocessing Queue)
    and pushes them to the Web Client (WebSockets).
    """
    global SYSTEM_STATE, status_queue
    print(">> [Dashboard] Status Monitor Bridge Active.")
    
    while True:
        try:
            if status_queue is None: 
                time.sleep(1)
                continue
                
            # Non-blocking get
            try:
                msg = status_queue.get(timeout=0.1)
                # DEBUG: Trace Data Flow
                # print(f">> [MONITOR] Got Msg: {msg.get('type')}", flush=True) 
                
                process_message(msg)
                
                # Push Update to Clients NOW
                socketio.emit('state_update', SYSTEM_STATE)
                # print(">> [MONITOR] Emitted State", flush=True)
                time.sleep(0) # Yield
                
            except queue.Empty:
                time.sleep(0.1)
                continue
                
        except (OSError, EOFError, BrokenPipeError):
            # Handle Manager death
            time.sleep(1)
        except Exception as e:
            print(f"[Monitor Error] {e}")
            time.sleep(1)

def safe_float(val):
    """Sanitize currency strings before casting to float."""
    if isinstance(val, (float, int)): return float(val)
    if isinstance(val, str):
        cleaned = val.replace('$', '').replace(',', '').strip()
        try:
            return float(cleaned)
        except ValueError:
            return 0.0
    return 0.0

def process_message(msg):
    global SYSTEM_STATE
    
    mtype = msg.get('type')
    SYSTEM_STATE['last_update'] = time.time()
    
    if mtype == 'log':
        raw_msg = msg.get('data')
        entry = {
            'time': datetime.now().strftime("%H:%M:%S"),
            'msg': str(raw_msg) if raw_msg is not None else "[EMPTY_LOG]"
        }
        SYSTEM_STATE['logs'].append(entry)
        if len(SYSTEM_STATE['logs']) > 100: SYSTEM_STATE['logs'].pop(0)
        
    elif mtype == 'agent_status':
        data = msg.get('data', {})
        SYSTEM_STATE['status'] = 'RUNNING' # Implicit
        if 'regime' in data: SYSTEM_STATE['regime'] = data['regime']
        if 'health' in data: SYSTEM_STATE['health'] = safe_float(data['health'])
        if 'balance' in data: SYSTEM_STATE['balance'] = safe_float(data['balance'])
        if 'equity' in data: SYSTEM_STATE['equity'] = safe_float(data['equity'])
        if 'pnl' in data: SYSTEM_STATE['pnl'] = safe_float(data['pnl'])
        if 'win_rate' in data: SYSTEM_STATE['win_rate'] = data['win_rate']
            
        # Chart Data
        if 'equity' in data:
            SYSTEM_STATE['equity_history'].append({
                't': datetime.now().strftime("%H:%M"),
                'y': safe_float(data['equity'])
            })
            if len(SYSTEM_STATE['equity_history']) > 100:
                SYSTEM_STATE['equity_history'].pop(0)
            
    elif mtype == 'positions':
        SYSTEM_STATE['positions'] = msg.get('data', [])
        
    elif mtype == 'radar':
        SYSTEM_STATE['radar'] = msg.get('data', [])

# --- Routes ---
@app.route('/')
def index():
    return "Holonic Trader API Gateway Active. Access UI at port 5173."

@app.route('/api/data') # Fallback for polling/initial load
def get_data():
    return jsonify(SYSTEM_STATE)

@app.route('/api/control', methods=['POST'])
def control_bot():
    global bot_process, status_queue, command_queue, stop_event
    
    data = request.json
    cmd = data.get('command')
    payload = data.get('data', {})
    
    print(f">> [Control] Received: {cmd}")
    
    if cmd == 'start':
        if run_bot is None:
             SYSTEM_STATE['logs'].append({'time': datetime.now().strftime("%H:%M:%S"), 'msg': f"ERROR: Cannot start. Missing deps: {dependency_error}"})
             return jsonify({'status': 'ERROR', 'msg': dependency_error})

        if bot_process and bot_process.is_alive():
             return jsonify({'status': 'ALREADY_RUNNING'})
             
        # Reset State
        SYSTEM_STATE['logs'] = []
        SYSTEM_STATE['status'] = 'STARTING...'
        
        # Drain Queues
        while not status_queue.empty(): status_queue.get()
        while not command_queue.empty(): command_queue.get()
        stop_event.clear()
        
        # Config
        run_cfg = {
            'symbol': payload.get('symbol', 'BTC/USDT'),
            'leverage': float(payload.get('leverage', 5.0)),
            'allocation': float(payload.get('allocation', 0.1))
        }
        
        # Start Process (Standard Multiprocessing)
        bot_process = multiprocessing.Process(
            target=run_bot, 
            args=(stop_event, status_queue, run_cfg, command_queue, DEBUG),
            daemon=True
        )
        bot_process.start()
        SYSTEM_STATE['status'] = 'RUNNING'
        return jsonify({'status': 'STARTED', 'pid': bot_process.pid})
        
    elif cmd == 'stop':
        if stop_event: stop_event.set()
        if command_queue: command_queue.put({'type': 'STOP'})
        
        SYSTEM_STATE['status'] = 'STOPPING...'
        # Wait a bit
        time.sleep(1)
        if bot_process and bot_process.is_alive():
             bot_process.terminate() # Hard kill if needed
        SYSTEM_STATE['status'] = 'STOPPED'
        return jsonify({'status': 'STOPPED'})
        
    elif cmd == 'panic':
        if command_queue: command_queue.put({'type': 'PANIC'})
        return jsonify({'status': 'PANIC_SENT'})

    return jsonify({'status': 'UNKNOWN'})

# --- Main Entry ---
if __name__ == '__main__':
    # Windows Multi-Processing Fix
    multiprocessing.freeze_support()
    
    print(">> [Dashboard] Initializing Shared Memory...")
    manager = multiprocessing.Manager()
    status_queue = manager.Queue()
    command_queue = manager.Queue()
    stop_event = manager.Event()
    
    # Start Monitor (Background Task)
    import threading
    monitor_thread = threading.Thread(target=status_monitor, daemon=True)
    monitor_thread.start()
        
    print(f">> 🚀 Web Engine (SocketIO/Threading) ready at http://localhost:{PORT}")
    try:
        # Use SocketIO Server 
        socketio.run(app, host=HOST, port=PORT, debug=DEBUG, use_reloader=False, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        print(">> Shutting down...")
        if bot_process: bot_process.terminate()
