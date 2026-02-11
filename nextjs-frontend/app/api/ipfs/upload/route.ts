import { NextRequest, NextResponse } from 'next/server';

const JS_BACKEND_URL = process.env.NEXT_PUBLIC_JS_BACKEND_URL || 'http://localhost:3001';

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();

    const response = await fetch(`${JS_BACKEND_URL}/ipfs/upload`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      return NextResponse.json(
        { success: false, error: errorData.error || 'IPFS upload failed' },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    const message = error instanceof Error ? error.message : 'IPFS upload failed';
    return NextResponse.json({ success: false, error: message }, { status: 500 });
  }
}
