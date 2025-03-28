# BUMPY - Background Removal Tool

BUMPY is a web application that allows users to easily remove backgrounds from images using advanced AI techniques.

## Features

- User authentication with Firebase
- Image background removal with rembg
- Dashboard to view processing history
- Responsive design for mobile and desktop
- Usage limits for free tier (25 images per day)

## Development Setup

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up environment variables in a `.env` file:
   ```
   FLASK_ENV=development
   FIREBASE_API_KEY=your_api_key
   FIREBASE_AUTH_DOMAIN=your_auth_domain
   FIREBASE_PROJECT_ID=your_project_id
   FIREBASE_STORAGE_BUCKET=your_storage_bucket
   FIREBASE_MESSAGING_SENDER_ID=your_messaging_sender_id
   FIREBASE_APP_ID=your_app_id
   FIREBASE_MEASUREMENT_ID=your_measurement_id
   ```
4. Run the development server:
   ```
   python app.py
   ```

## Production Deployment on Render

1. Fork or clone this repository to your GitHub account.
2. Create a new Web Service on Render.
3. Connect your GitHub repository.
4. Use the following settings:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app`
5. Add the required environment variables in the Render dashboard.
6. Deploy the service.

## Environment Variables

- `FLASK_ENV`: Set to 'development' or 'production'
- `PORT`: Port for the application to run on (set by Render in production)
- `FIREBASE_*`: Firebase configuration variables
- `FIREBASE_SERVICE_ACCOUNT_KEY`: JSON string of Firebase service account key (for production)

## Technologies Used

- Flask (Python web framework)
- Firebase (Authentication and data storage)
- rembg (Background removal library)
- Tailwind CSS (Styling)
- Alpine.js (Frontend interactivity)

## License

MIT 