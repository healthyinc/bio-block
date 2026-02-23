import { NextRequest, NextResponse } from 'next/server';

const JS_BACKEND_URL = process.env.NEXT_PUBLIC_JS_BACKEND_URL || 'http://localhost:3001';
const PYTHON_BACKEND_URL = process.env.NEXT_PUBLIC_PYTHON_BACKEND_URL || 'http://localhost:3002';

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const file = formData.get('file') as File | null;

    if (!file) {
      return NextResponse.json({ error: 'No file provided' }, { status: 400 });
    }

    // Determine backend based on file type
    const isImage = file.type.startsWith('image/');
    const backendUrl = isImage ? PYTHON_BACKEND_URL : JS_BACKEND_URL;
    const endpoint = isImage ? '/anonymize_image' : '/anonymize';

    const response = await fetch(`${backendUrl}${endpoint}`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      return NextResponse.json(
        { error: errorData.detail || errorData.error || 'Anonymization failed' },
        { status: response.status }
      );
    }

    // Check if response is JSON (multi-file) or blob (single file)
    const contentType = response.headers.get('content-type');
    if (contentType?.includes('application/json')) {
      const data = await response.json();
      return NextResponse.json(data);
    } else {
      const blob = await response.blob();
      return new NextResponse(blob, {
        status: 200,
        headers: {
          'Content-Type': response.headers.get('content-type') || 'application/octet-stream',
          'Content-Disposition': response.headers.get('content-disposition') || '',
        },
      });
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Anonymization failed';
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
