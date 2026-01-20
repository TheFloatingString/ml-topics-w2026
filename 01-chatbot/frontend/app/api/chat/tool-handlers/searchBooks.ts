interface GutenbergBook {
  id: number;
  title: string;
  authors: { name: string }[];
}

interface BookResult {
  id: number;
  title: string;
  author_name: string | null;
  publisher: string | null;
  edition: string | null;
  year_of_publication: number | null;
}

export async function searchBooks(args: { query: string }) {
  // Query the Gutenberg API for books matching the search
  const response = await fetch(
    `https://gutendex.com/books/?search=${encodeURIComponent(args.query)}`
  );
  const data = await response.json();

  const books: BookResult[] =
    data.results?.map((book: GutenbergBook) => ({
      id: book.id,
      title: book.title,
      author_name: book.authors?.[0]?.name || null,
      publisher: null, // Gutenberg API doesn't provide publisher info
      edition: null, // Gutenberg API doesn't provide edition info
      year_of_publication: null, // Gutenberg API doesn't provide publication year
    })) || [];

  return { books };
}
