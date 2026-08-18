import { NextResponse } from 'next/server';

const JS_BACKEND_URL = process.env.NEXT_PUBLIC_JS_BACKEND_URL || 'http://localhost:3001';

export async function GET() {
  try {
    const response = await fetch(`${JS_BACKEND_URL}/api/health`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
    });

    if (!response.ok) {
      return NextResponse.json(
        { success: false, message: `Backend returned status ${response.status}` },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json({ success: true, message: 'Backend connected', data });
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Failed to connect to backend';
    return NextResponse.json({ success: false, message }, { status: 503 });
  }
}
