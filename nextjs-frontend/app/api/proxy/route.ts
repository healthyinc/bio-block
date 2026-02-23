import { NextRequest, NextResponse } from 'next/server';

const JS_BACKEND_URL = process.env.NEXT_PUBLIC_JS_BACKEND_URL || 'http://localhost:3001';
const PYTHON_BACKEND_URL = process.env.NEXT_PUBLIC_PYTHON_BACKEND_URL || 'http://localhost:3002';

export async function POST(request: NextRequest) {
  try {
    const { url, method = 'GET', body, headers: customHeaders } = await request.json();

    if (!url) {
      return NextResponse.json({ error: 'URL is required' }, { status: 400 });
    }

    // Determine full URL
    let fullUrl = url;
    if (url.startsWith('/js/')) {
      fullUrl = `${JS_BACKEND_URL}${url.replace('/js', '')}`;
    } else if (url.startsWith('/py/')) {
      fullUrl = `${PYTHON_BACKEND_URL}${url.replace('/py', '')}`;
    }

    const fetchOptions: RequestInit = {
      method,
      headers: {
        'Content-Type': 'application/json',
        ...customHeaders,
      },
    };

    if (body && method !== 'GET') {
      fetchOptions.body = JSON.stringify(body);
    }

    const response = await fetch(fullUrl, fetchOptions);

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      return NextResponse.json(
        { error: errorData.detail || errorData.error || 'Proxy request failed' },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Proxy request failed';
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
