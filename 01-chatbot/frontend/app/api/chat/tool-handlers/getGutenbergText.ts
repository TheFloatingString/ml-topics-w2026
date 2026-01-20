export async function getGutenbergText(args: { id: number }) {
  const id = args.id || 1497;
  const response = await fetch(
    `https://www.gutenberg.org/cache/epub/${id}/pg${id}.txt`
  );
  const text = await response.text();
  return { text };
}
