import { NextResponse } from "next/server";
import { listSamples } from "@/lib/db";

export async function GET() {
  try {
    const samples = await listSamples();
    return NextResponse.json(samples);
  } catch (e) {
    console.error(e);
    return new NextResponse("Failed", { status: 500 });
  }
}
