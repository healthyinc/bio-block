import { ethers } from 'ethers';

class AuthService {
    constructor() {
        this.token = localStorage.getItem('authToken');
        this.walletAddress = localStorage.getItem('walletAddress');
    }

    async authenticateWallet(walletAddress) {
        try {
            const message = `Bio-Block Authentication\nWallet: ${walletAddress}\nTimestamp: ${Date.now()}`;
            
            const provider = new ethers.BrowserProvider(window.ethereum);
            const signer = await provider.getSigner();
            const signature = await signer.signMessage(message);

            const response = await fetch('/api/auth/wallet', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    walletAddress,
                    signature,
                    message
                })
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || 'Authentication failed');
            }

            const result = await response.json();
            
            this.token = result.token;
            this.walletAddress = result.walletAddress;
            localStorage.setItem('authToken', this.token);
            localStorage.setItem('walletAddress', this.walletAddress);

            return result;

        } catch (error) {
            console.error('Wallet authentication error:', error);
            throw error;
        }
    }

    isAuthenticated() {
        return !!this.token && !!this.walletAddress;
    }

    getAuthHeaders() {
        if (!this.token) {
            throw new Error('Not authenticated');
        }
        
        return {
            'Authorization': `Bearer ${this.token}`,
            'Content-Type': 'application/json'
        };
    }

    logout() {
        this.token = null;
        this.walletAddress = null;
        localStorage.removeItem('authToken');
        localStorage.removeItem('walletAddress');
    }

    getWalletAddress() {
        return this.walletAddress;
    }

    async verifyToken() {
        if (!this.token) {
            return false;
        }

        try {
            const response = await fetch('/api/health', {
                headers: this.getAuthHeaders()
            });
            
            return response.ok;
        } catch (error) {
            console.error('Token verification failed:', error);
            return false;
        }
    }
}

const authService = new AuthService();

export default authService;