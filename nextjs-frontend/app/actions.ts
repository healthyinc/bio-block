'use server';

interface SearchParams {
  query: string;
  filters?: {
    disease?: string;
    dataType?: string;
    gender?: string;
    dataSource?: string;
    minPrice?: string;
    maxPrice?: string;
  };
}

interface SearchResult {
  cid: string;
  summary: string;
  dataset_title: string;
  metadata: Record<string, unknown>;
  similarity_score: number;
}

interface SearchResponse {
  success: boolean;
  results?: SearchResult[];
  error?: string;
}

export async function searchDocuments(params: SearchParams): Promise<SearchResponse> {
  const PYTHON_BACKEND_URL = process.env.NEXT_PUBLIC_PYTHON_BACKEND_URL || 'http://localhost:3002';

  try {
    const response = await fetch(`${PYTHON_BACKEND_URL}/search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      return { success: false, error: errorData.detail || 'Search failed' };
    }

    const data = await response.json();
    return { success: true, results: data.results || data };
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Search request failed';
    return { success: false, error: message };
  }
}

interface StoreParams {
  summary: string;
  dataset_title: string;
  cid: string;
  metadata: Record<string, unknown>;
}

interface StoreResponse {
  success: boolean;
  error?: string;
}

export async function storeMetadata(params: StoreParams): Promise<StoreResponse> {
  const PYTHON_BACKEND_URL = process.env.NEXT_PUBLIC_PYTHON_BACKEND_URL || 'http://localhost:3002';

  try {
    const response = await fetch(`${PYTHON_BACKEND_URL}/store`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      return { success: false, error: errorData.detail || 'Store failed' };
    }

    return { success: true };
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Store request failed';
    return { success: false, error: message };
  }
}

interface HealthResponse {
  success: boolean;
  message: string;
  data?: Record<string, unknown>;
}

export async function checkHealth(): Promise<HealthResponse> {
  const JS_BACKEND_URL = process.env.NEXT_PUBLIC_JS_BACKEND_URL || 'http://localhost:3001';

  try {
    const response = await fetch(`${JS_BACKEND_URL}/api/health`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
    });

    if (!response.ok) {
      return { success: false, message: `Backend returned status ${response.status}` };
    }

    const data = await response.json();
    return { success: true, message: 'Backend connected', data };
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Health check failed';
    return { success: false, message };
  }
}
