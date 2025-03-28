// Auth.js - Handles Firebase authentication

document.addEventListener('DOMContentLoaded', function() {
    // Get elements
    const authButtons = document.getElementById('auth-buttons');
    const userMenu = document.getElementById('user-menu');
    const userEmail = document.getElementById('user-email');
    const userAvatar = document.getElementById('user-avatar');
    const profilePic = document.getElementById('profile-pic');
    const userInitials = document.getElementById('user-initials');
    const userDropdownBtn = document.getElementById('user-dropdown-btn');
    const userDropdown = document.getElementById('user-dropdown');
    const logoutBtn = document.getElementById('logout-btn');
    
    // Function to toggle dropdown
    function toggleDropdown() {
        if (userDropdown.classList.contains('hidden')) {
            userDropdown.classList.remove('hidden');
            userDropdown.classList.add('menu-fade-in');
        } else {
            userDropdown.classList.add('hidden');
            userDropdown.classList.remove('menu-fade-in');
        }
    }
    
    // Set up user avatar (profile pic or initials)
    function setupUserAvatar(user) {
        if (user.photoURL) {
            // If user has a profile picture, use it
            profilePic.src = user.photoURL;
            profilePic.classList.remove('hidden');
            userInitials.classList.add('hidden');
        } else {
            // Otherwise, show initials on a colored background
            profilePic.classList.add('hidden');
            userInitials.classList.remove('hidden');
            
            // Get user initials
            let initials = "U";
            if (user.displayName) {
                const names = user.displayName.split(' ');
                if (names.length >= 2) {
                    initials = names[0][0] + names[1][0];
                } else if (names.length === 1 && names[0].length > 0) {
                    initials = names[0][0];
                }
            } else if (user.email) {
                initials = user.email[0].toUpperCase();
            }
            
            userInitials.textContent = initials;
            
            // Random background color based on user ID or email
            const seed = user.uid || user.email || "default";
            const hash = seed.split('').reduce((acc, char) => {
                return char.charCodeAt(0) + ((acc << 5) - acc);
            }, 0);
            
            const h = Math.abs(hash) % 360;
            userAvatar.style.backgroundColor = `hsl(${h}, 70%, 80%)`;
        }
    }
    
    // Check if user is logged in
    firebase.auth().onAuthStateChanged(function(user) {
        if (user) {
            // User is signed in
            userEmail.textContent = user.email;
            setupUserAvatar(user);
            
            authButtons.classList.add('hidden');
            userMenu.classList.remove('hidden');
            
            console.log("User signed in:", user.email);
        } else {
            // User is signed out
            authButtons.classList.remove('hidden');
            userMenu.classList.add('hidden');
            
            console.log("No user signed in");
        }
    });
    
    // Toggle dropdown when button is clicked
    if (userDropdownBtn) {
        userDropdownBtn.addEventListener('click', function(e) {
            e.stopPropagation();
            toggleDropdown();
        });
    }
    
    // Close dropdown when clicking outside
    document.addEventListener('click', function(e) {
        if (userDropdown && !userDropdownBtn.contains(e.target) && !userDropdown.contains(e.target)) {
            userDropdown.classList.add('hidden');
        }
    });
    
    // Sign out user
    if (logoutBtn) {
        logoutBtn.addEventListener('click', function(e) {
            e.preventDefault();
            
            // Add animation before signing out
            userMenu.classList.add('animate-fade-out');
            
            setTimeout(() => {
                firebase.auth().signOut().then(function() {
                    // Sign-out successful
                    window.location.href = '/';
                }).catch(function(error) {
                    // An error happened
                    console.error("Error signing out:", error);
                });
            }, 300);
        });
    }
}); 