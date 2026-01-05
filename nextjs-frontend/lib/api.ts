const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:3001';

interface BackendHealthResponse {
  success: boolean;
  message: string;
  data?: {
    status: string;
    timestamp: string;
  };
}

/**
 * Check backend health/connectivity
 * @returns Object with success status and message
 */
export const checkBackendHealth = async (): Promise<BackendHealthResponse> => {
  try {
    const response = await fetch(`${BACKEND_URL}/api/health`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      return {
        success: false,
        message: `Backend returned status ${response.status}`,
      };
    }

    const data = await response.json();
    return {
      success: true,
      message: 'Backend connected successfully',
      data,
    };
  } catch (error) {
    console.error('Backend health check failed:', error);
    return {
      success: false,
      message: error instanceof Error ? error.message : 'Failed to connect to backend',
    };
  }
};

/**
 * Get backend URL
 * @returns Backend URL
 */
export const getBackendUrl = (): string => {
  return BACKEND_URL;
};
