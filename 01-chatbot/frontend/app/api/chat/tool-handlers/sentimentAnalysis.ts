import natural from "natural";

const Analyzer = natural.SentimentAnalyzer;
const stemmer = natural.PorterStemmer;
const analyzer = new Analyzer("English", stemmer, "afinn");

export function analyzeSentiment(args: { sentences: string[] }) {
  const { sentences } = args;

  const results = sentences.map((sentence) => {
    const tokens = new natural.WordTokenizer().tokenize(sentence) || [];
    const score = analyzer.getSentiment(tokens);

    let sentiment: "positive" | "negative" | "neutral";
    if (score > 0.05) {
      sentiment = "positive";
    } else if (score < -0.05) {
      sentiment = "negative";
    } else {
      sentiment = "neutral";
    }

    return { sentence, score, sentiment };
  });

  const sentiments = results.map((r) => r.sentiment);

  return {
    results,
    sentiments,
    breakdown: {
      positive: sentiments.filter((s) => s === "positive").length,
      negative: sentiments.filter((s) => s === "negative").length,
      neutral: sentiments.filter((s) => s === "neutral").length,
    },
  };
}
