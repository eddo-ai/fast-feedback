# FAST Feedback Client

This Next.js application provides the web interface for the FAST project, which helps educators analyze student work and generate feedback.

## Development

Use `pnpm` to install dependencies and start the development server:

```bash
pnpm install
pnpm dev
```

## Linting

Run lint checks with:

```bash
pnpm lint
```

For more commands and contribution guidelines, see [AGENTS.md](../AGENTS.md).

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.

## Environment Variables

Create a `.env.local` file at the project root with the following variables:

```bash
POSTGRES_URL=postgres://user:password@localhost:5432/eddo
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your-anon-key
```

- `SUPABASE_URL` and `SUPABASE_ANON_KEY` are used if you want to connect to Supabase from the client or server. You can find these in your Supabase project dashboard under Project Settings > API.

Run the development server with `pnpm dev` and open http://localhost:3000.
