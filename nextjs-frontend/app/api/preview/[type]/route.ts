import { NextRequest, NextResponse } from 'next/server';

const JS_BACKEND_URL = process.env.NEXT_PUBLIC_JS_BACKEND_URL || 'http://localhost:3001';

export async function POST(
  request: NextRequest,
  { params }: { params: Promise<{ type: string }> }
) {
  try {
    const { type } = await params;
    const validTypes = ['image', 'spreadsheet', 'pdf', 'dicom'];

    if (!validTypes.includes(type)) {
      return NextResponse.json(
        { error: `Invalid preview type: ${type}. Valid types: ${validTypes.join(', ')}` },
        { status: 400 }
      );
    }

    const formData = await request.formData();

    const response = await fetch(`${JS_BACKEND_URL}/api/preview/${type}`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      return NextResponse.json(
        { error: errorData.error || 'Preview generation failed' },
        { status: response.status }
      );
    }

    const contentType = response.headers.get('content-type');

    // Spreadsheet returns JSON, others return blobs
    if (type === 'spreadsheet' || contentType?.includes('application/json')) {
      const data = await response.json();
      return NextResponse.json(data);
    } else {
      const blob = await response.blob();
      return new NextResponse(blob, {
        status: 200,
        headers: {
          'Content-Type': contentType || 'application/octet-stream',
        },
      });
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Preview generation failed';
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
