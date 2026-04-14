"""
Retail Inventory Twin — Dashboard App
Run with: python app.py
Access on this machine : http://127.0.0.1:5001
Access from teammates  : http://<this-machine-IP>:5001
"""

from flask import Flask, render_template, jsonify, abort
import pandas as pd
from datetime import datetime, timedelta
import socket
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, 'templates'))
app.config['TEMPLATES_AUTO_RELOAD'] = True

CSV_PATH = os.path.join(BASE_DIR, 'stockout_results.csv')


def load_stockout():
    return pd.read_csv(CSV_PATH)


@app.after_request
def no_cache(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    return response


# ── Helpers ──────────────────────────────────────────────────────

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


# ── Routes ───────────────────────────────────────────────────────

@app.route('/')
def landing():
    """Retailer selection landing page."""
    df = load_stockout()
    summary = []
    for rid in sorted(df['retailer_id'].unique()):
        rdf = df[df['retailer_id'] == rid]
        summary.append({
            'retailer_id': rid,
            'total':    len(rdf),
            'critical': int((rdf['alert_level'] == 'CRITICAL').sum()),
            'warning':  int((rdf['alert_level'] == 'WARNING').sum()),
            'safe':     int((rdf['alert_level'] == 'SAFE').sum()),
        })
    return render_template('landing.html', retailers=summary)


@app.route('/retailer/<retailer_id>')
def dashboard(retailer_id):
    """Per-retailer dashboard page."""
    df = load_stockout()
    valid = df['retailer_id'].unique().tolist()
    if retailer_id not in valid:
        abort(404)
    return render_template('dashboard.html', retailer_id=retailer_id)


# ── API ───────────────────────────────────────────────────────────

@app.route('/api/stockout/<retailer_id>')
def api_stockout(retailer_id):
    df = load_stockout()
    rdf = df[df['retailer_id'] == retailer_id].copy()
    if rdf.empty:
        abort(404)

    alert_order = {'CRITICAL': 0, 'WARNING': 1, 'SAFE': 2}
    rdf['_rank'] = rdf['alert_level'].map(alert_order)
    rdf = rdf.sort_values(['_rank', 'days_until_stockout']).drop(columns=['_rank'])

    summary = {
        'total':    len(rdf),
        'critical': int((rdf['alert_level'] == 'CRITICAL').sum()),
        'warning':  int((rdf['alert_level'] == 'WARNING').sum()),
        'safe':     int((rdf['alert_level'] == 'SAFE').sum()),
    }
    return jsonify({'summary': summary, 'items': rdf.to_dict(orient='records')})


@app.route('/api/purchase_order/<retailer_id>')
def api_purchase_order(retailer_id):
    df = load_stockout()
    rdf = df[df['retailer_id'] == retailer_id].copy()
    if rdf.empty:
        abort(404)

    po_items = rdf[
        rdf['alert_level'].isin(['CRITICAL', 'WARNING']) &
        (rdf['recommended_order_qty'] > 0)
    ].copy()

    alert_order = {'CRITICAL': 0, 'WARNING': 1, 'SAFE': 2}
    po_items['_rank'] = po_items['alert_level'].map(alert_order)
    po_items = po_items.sort_values(['_rank', 'days_until_stockout']).drop(columns=['_rank'])

    lead_time = timedelta(days=5)
    print(f"DEBUG: po_items count={len(po_items)}, using new order_date logic", flush=True)
    if not po_items.empty:
        earliest_stockout = pd.to_datetime(po_items['stockout_date']).min()
        order_date = earliest_stockout - lead_time
        expected_delivery = order_date + lead_time
    else:
        order_date = datetime.today()
        expected_delivery = order_date + lead_time

    return jsonify({
        'po_number':        f"PO-{retailer_id}-{order_date.strftime('%Y%m%d')}",
        'retailer_id':      retailer_id,
        'order_date':       order_date.strftime('%Y-%m-%d'),
        'expected_delivery': expected_delivery.strftime('%Y-%m-%d'),
        'items':            po_items.to_dict(orient='records'),
        'total_line_items': len(po_items),
        'total_units':      int(po_items['recommended_order_qty'].sum()),
    })


# ── Entry point ───────────────────────────────────────────────────

if __name__ == '__main__':
    if not os.path.exists(CSV_PATH):
        print("ERROR: stockout_results.csv not found. Run generate_stockout_results.py first.")
    else:
        local_ip = get_local_ip()
        print("\n" + "="*55)
        print("  Retail Inventory Twin — Dashboard")
        print("="*55)
        print(f"  Local  : http://127.0.0.1:5001")
        print(f"  Network: http://{local_ip}:5001   <-- share with teammates")
        print("="*55 + "\n")
        app.run(host='0.0.0.0', port=8080, debug=False)
