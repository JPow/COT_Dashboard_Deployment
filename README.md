# COT Dashboard Deployment

A web-based dashboard for visualizing Commitment of Traders (COT) data, featuring interactive charts and real-time updates.

## Features

- Interactive price and open interest analysis
- Retail vs Commercial positioning visualization
- Open Interest Index tracking
- Mobile-responsive design
- Real-time data updates

## Installation

1. Clone the repository:
```bash
git clone https://github.com/JPow/COT_Dashboard_Deployment.git
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the dashboard:
```bash
python app.py
```

2. Access the dashboard at `http://localhost:8050`

## Dependencies

- dash==2.16.1
- cot_reports==0.1.3
- pandas==2.1.4
- plotly==5.9.0
- dash-bootstrap-components==1.6.0
- numpy==1.23.5
- gunicorn==21.2.0

## Deployment

The dashboard is deployed using Render. Access it at: [Your Render URL]
Remember to update the price date for the most recent price data. 

## License

GPL-3.0 license
