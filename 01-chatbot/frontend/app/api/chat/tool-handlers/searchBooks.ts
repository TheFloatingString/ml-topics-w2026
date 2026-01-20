export async function searchBooks(args: { query: string }) {
  // Query the Gutenberg API for books matching the search
  const response = await fetch(
    `https://gutendex.com/books/?search=${encodeURIComponent(args.query)}`
  );
  const data = await response.json();

  const book_ids = data.results?.map((book: { id: number }) => book.id) || [];

  return { book_ids };
}
