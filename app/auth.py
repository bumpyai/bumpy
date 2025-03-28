from flask import Blueprint, render_template, request, redirect, url_for, flash, session, jsonify, current_app
import firebase_admin
from firebase_admin import auth
import json
import os

auth_bp = Blueprint('auth', __name__, url_prefix='/auth')

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')
    
    return render_template('login.html')

@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'GET':
        return render_template('register.html')
    
    return render_template('register.html')

@auth_bp.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('user_email', None)
    return redirect(url_for('index'))

@auth_bp.route('/verify-token', methods=['POST'])
def verify_token():
    try:
        id_token = request.json.get('idToken')
        
        # Development mode bypass
        is_development = os.environ.get('FLASK_ENV') == 'development'
        if is_development and id_token == 'fake-token-for-development':
            # In development, accept the fake token
            uid = 'dev-user-123'
            email = 'dev@example.com'
            
            # Store user info in session
            session['user_id'] = uid
            session['user_email'] = email
            
            return jsonify({'success': True, 'uid': uid, 'email': email})
            
        try:
            # Verify with Firebase Admin SDK
            decoded_token = auth.verify_id_token(id_token)
            uid = decoded_token['uid']
            email = decoded_token.get('email', '')
        except Exception as e:
            if is_development:
                # In development, create a fake user if token verification fails
                print(f"Token verification failed in development mode: {e}")
                uid = 'dev-user-123'
                email = 'dev@example.com'
            else:
                # In production, re-raise the exception
                raise
        
        # Store user info in session
        session['user_id'] = uid
        session['user_email'] = email
        
        return jsonify({'success': True, 'uid': uid, 'email': email})
    except Exception as e:
        print(f"Token verification error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 401 