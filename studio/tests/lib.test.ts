import assert from "node:assert/strict";
import { test } from "node:test";
import { checkAction } from "../src/lib/actions.ts";
import { newSession, passwordOk, sameText, sessionValid } from "../src/lib/session.ts";

process.env.DASHBOARD_PASSWORD = "test-password";
process.env.INGEST_TOKEN = "test-ingest";

test("session cookie: valid, expired, tampered, key rotation", async () => {
  const t = await newSession();
  assert.ok(t && (await sessionValid(t)));
  assert.equal(await sessionValid(t!.replace(/.$/, (c) => (c === "A" ? "B" : "A"))), false);
  assert.equal(await sessionValid(await newSession(Date.now() - 40 * 86400_000)), false);
  assert.equal(await sessionValid("hs1.9999999999"), false);
  process.env.DASHBOARD_PASSWORD = "changed";
  assert.equal(await sessionValid(t), false); // a new password signs every device out
  process.env.DASHBOARD_PASSWORD = "test-password";
});

test("password check", async () => {
  assert.ok(await passwordOk("test-password"));
  assert.equal(await passwordOk("12345"), false);
  assert.equal(await passwordOk(""), false);
  assert.ok(sameText("abc", "abc") && !sameText("abc", "abd") && !sameText("ab", "abc"));
});

test("write validation", () => {
  assert.equal(checkAction({ kind: "approve", video: "pip_pilot_v2" }).ok, true);
  assert.equal(checkAction({ kind: "needs_changes", video: "pip_pilot_v2" }).ok, false); // note required
  assert.equal(checkAction({ kind: "needs_changes", video: "pip_pilot_v2", note: "slower" }).ok, true);
  assert.equal(checkAction({ kind: "approve", video: "../etc" }).ok, false);
  assert.equal(checkAction({ kind: "delete_everything", video: "x" }).ok, false);
  assert.equal(checkAction({ kind: "queue_story", story: "pip", mode: "full" }).ok, true);
  assert.equal(checkAction({ kind: "queue_story", story: "pip", mode: "huge" }).ok, false);
  assert.equal(checkAction({ kind: "published", video: "a", url: "javascript:alert(1)" }).ok, false);
  const r = checkAction({ kind: "approve", video: "a", note: "x".repeat(5000) });
  assert.ok(r.ok && r.action.note!.length === 2000);
});
