import { Console } from "@/components/Console";
import { readLive, readSnapshot } from "@/lib/store";
import type { SnapshotResponse } from "@/lib/types";

/** Render time handed to the client so its first render matches the server HTML. */
function serverTime(): number {
  return Date.now();
}

export const dynamic = "force-dynamic";

export default async function Home() {
  let initial: SnapshotResponse | null = null;
  try {
    const [{ snapshot, storedAt }, live] = await Promise.all([readSnapshot(), readLive().catch(() => ({ url: null, at: null }))]);
    initial = { snapshot, stored_at: storedAt, live };
  } catch {
    initial = null; // the client retries
  }
  return <Console initial={initial} serverNow={serverTime()} />;
}
