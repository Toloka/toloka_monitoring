from toloka_monitoring.api.app import create_app

if __name__ == '__main__':
    app = create_app()
    app.run(port=8000, host="0.0.0.0", debug=False)