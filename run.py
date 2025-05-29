#!/usr/bin/env python
import os
import sys
import subprocess

def create_directories():
    """Create necessary directories if they don't exist"""
    os.makedirs('templates', exist_ok=True)
    os.makedirs('dashboards', exist_ok=True)

def install_dependencies():
    """Install required dependencies"""
    print("Installing required dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def setup_environment():
    """Setup the environment for the application"""
    create_directories()
    if not os.path.exists('.venv'):
        print("Creating virtual environment...")
        subprocess.check_call([sys.executable, "-m", "venv", ".venv"])
        
        if sys.platform == 'win32':
            pip_path = os.path.join('.venv', 'Scripts', 'pip')
        else:
            pip_path = os.path.join('.venv', 'bin', 'pip')
            
        subprocess.check_call([pip_path, "install", "-r", "requirements.txt"])
    
def run_app():
    """Run the Flask application"""
    try:
        from app import app
        print("Starting NBA Player Rankings website...")
        print("Visit http://127.0.0.1:5000/ in your browser to view the rankings")
        app.run(debug=True)
    except ImportError:
        print("Error: Could not import Flask application. Make sure all dependencies are installed.")
        sys.exit(1)

if __name__ == "__main__":
    create_directories()
    
    # Check if running in a virtual environment
    if not hasattr(sys, 'real_prefix') and not hasattr(sys, 'base_prefix') != sys.prefix:
        # Not in a virtual environment, try to activate or create one
        if os.path.exists('.venv'):
            print("Virtual environment found. Installing/updating dependencies...")
            install_dependencies()
        else:
            print("Setting up environment...")
            setup_environment()
    
    # Run the app
    run_app() 