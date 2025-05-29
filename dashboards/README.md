# NBA Player Rankings Website

This simple web application displays NBA player rankings based on predicted RAPTOR ratings.

## Features

- View the top NBA players based on total RAPTOR ratings
- Interactive bar chart visualizing offensive and defensive contributions
- Filter by different prediction years (2022, 2025, etc.)
- Customize the number of top players to display (10, 25, 50, or 100)

## Running the Application

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Flask application:
```bash
python app.py
```

3. Open your browser and navigate to:
```
http://127.0.0.1:5000/
```

## Technology Stack

- Backend: Flask (Python)
- Frontend: HTML, CSS, JavaScript, Bootstrap 5
- Data Visualization: Chart.js
- Data Processing: Pandas

## Understanding the Metrics

- **Offensive RAPTOR**: Measures a player's impact on offensive performance
- **Defensive RAPTOR**: Measures a player's impact on defensive performance
- **Total RAPTOR**: Combined metric to assess overall player impact

## Future Improvements

- Add player search functionality
- Add position filtering
- Include player photos
- Compare players side-by-side 