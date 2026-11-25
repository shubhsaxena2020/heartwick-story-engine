import { Mark } from "@/components/Mark";

const ERRORS: Record<string, string> = {
  wrong: "That password is not right. Try again.",
  throttled: "Too many attempts from this network. Wait 10 minutes, then try again.",
  config: "The server has no DASHBOARD_PASSWORD set. Add it in the Vercel project settings.",
};

export default async function LoginPage({ searchParams }: { searchParams: Promise<{ e?: string; next?: string }> }) {
  const { e, next } = await searchParams;
  const error = e ? ERRORS[e] : null;
  return (
    <main className="relative z-10 grid min-h-dvh place-items-center px-4">
      <form method="post" action="/api/login" className="panel bracket page-in w-full max-w-[23.75rem] p-6 sm:p-8">
        <div className="mb-8 flex items-center gap-3">
          <Mark size={36} />
          <div>
            <div className="display text-[2.125rem] font-extrabold uppercase">Heartwick</div>
            <div className="label mt-1">studio console · restricted</div>
          </div>
        </div>
        <label htmlFor="pw" className="label mb-2 block">
          Password
        </label>
        <input
          id="pw"
          name="password"
          type="password"
          required
          autoFocus
          autoComplete="current-password"
          className="w-full rounded-[3px] border border-rule-strong bg-bg px-3 py-2.5 text-[0.875rem] text-ink outline-none focus:border-sodium"
          aria-describedby={error ? "pw-err" : undefined}
        />
        <input type="hidden" name="next" value={next ?? "/"} />
        {error && (
          <p id="pw-err" role="alert" className="mt-3 text-[0.75rem] text-err">
            {error}
          </p>
        )}
        <button
          type="submit"
          className="mt-6 w-full rounded-[3px] bg-sodium px-3 py-2.5 text-[0.75rem] font-bold uppercase tracking-[0.14em] text-[#1a0d04] hover:brightness-110"
        >
          Open studio
        </button>
        <p className="mt-5 text-[0.6875rem] leading-relaxed text-faint">Session lasts 30 days on this device.</p>
      </form>
    </main>
  );
}
