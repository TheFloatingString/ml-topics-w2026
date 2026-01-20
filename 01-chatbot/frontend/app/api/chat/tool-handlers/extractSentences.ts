import natural from "natural";

const tokenizer = new natural.SentenceTokenizer();

export function extractSentencesWithKeyword(args: { text: string; keyword: string }) {
  const { text, keyword } = args;
  const keywordLower = keyword.toLowerCase();

  // Split into paragraphs first, then only tokenize paragraphs containing the keyword
  // This is much faster than tokenizing the entire book
  // Handle both Unix (\n) and Windows (\r\n) line endings
  const paragraphs = text.split(/\r?\n\s*\r?\n/);
  const relevantParagraphs = paragraphs.filter((p) =>
    p.toLowerCase().includes(keywordLower)
  );

  // Only tokenize the relevant paragraphs
  const sentences: string[] = [];
  for (const para of relevantParagraphs) {
    const paraSentences = tokenizer.tokenize(para);
    if (!paraSentences) continue;
    for (const sentence of paraSentences) {
      const trimmed = sentence.trim();
      if (trimmed.toLowerCase().includes(keywordLower)) {
        sentences.push(trimmed);
      }
    }
  }

  return { sentences };
}
