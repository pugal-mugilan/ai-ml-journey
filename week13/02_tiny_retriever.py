import numpy as np
from sentence_transformers import SentenceTransformer

class TinyRetriever:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.corpus_embeddings = None
        self.corpus_texts = None

    def index(self, texts):
        self.corpus_texts = texts
        self.corpus_embeddings = self.model.encode(
            texts, normalize_embeddings=True
        )

    def query(self, q, k=3):
        q_emb = self.model.encode([q], normalize_embeddings=True)[0]
        scores = self.corpus_embeddings @ q_emb
        top_k_idx = np.argsort(-scores)[:k]
        return [(self.corpus_texts[i], float(scores[i])) for i in top_k_idx]


corpus = [
    # Cooking (17 sentences)
    "Dice the onions finely before adding them to the hot pan",
    "Marinate the chicken in yogurt and spices for at least two hours",
    "A pinch of salt enhances the sweetness in chocolate cake recipes",
    "Stir the risotto continuously to release the starch from the rice",
    "Roast the vegetables at 200 degrees until they caramelize",
    "Fresh basil leaves should be added at the end to preserve flavor",
    "Knead the dough for ten minutes until it becomes smooth and elastic",
    "Use a meat thermometer to check if the steak is medium rare",
    "Simmer the tomato sauce on low heat for a richer taste",
    "Blanch the broccoli in boiling water then transfer to ice bath",
    "Season the cast iron skillet with oil after every wash",
    "Whisk the eggs vigorously to incorporate air into the batter",
    "Toast the cumin seeds in a dry pan until they become fragrant",
    "Deglaze the pan with white wine to create a flavorful sauce",
    "Let the bread dough rise in a warm place for one hour",
    "Crush garlic cloves with the flat side of a knife before mincing",
    "A slow cooker makes tender pulled pork with minimal effort",

    # Coding (17 sentences)
    "Use list comprehensions in Python for cleaner and faster loops",
    "Git rebase rewrites commit history to create a linear timeline",
    "A dictionary lookup in Python runs in O(1) average time complexity",
    "Write unit tests before refactoring to catch regressions early",
    "Async await in JavaScript handles non-blocking IO operations",
    "Docker containers share the host kernel unlike virtual machines",
    "Use environment variables to store secrets instead of hardcoding them",
    "A REST API uses HTTP methods like GET POST PUT and DELETE",
    "Recursion needs a base case to prevent infinite function calls",
    "Type hints in Python improve code readability and IDE support",
    "SQL joins combine rows from two tables based on related columns",
    "A binary search runs in O(log n) by halving the search space",
    "Version pinning in requirements.txt prevents dependency conflicts",
    "Use try except blocks to handle exceptions gracefully in Python",
    "Kubernetes orchestrates container deployment scaling and management",
    "A linked list allows O(1) insertion at the head of the sequence",
    "Pytest fixtures provide reusable setup code for test functions",

    # Weather (16 sentences)
    "Heavy rainfall is expected across the southern coast this weekend",
    "The temperature will drop below freezing overnight in the mountains",
    "A warm front moving eastward will bring humid conditions tomorrow",
    "UV index is extremely high today so apply sunscreen before going out",
    "Fog advisory issued for the valley region until mid morning",
    "Strong winds exceeding 60 mph may cause power outages in the area",
    "The monsoon season typically begins in June and lasts until September",
    "Barometric pressure is falling which often signals incoming storms",
    "Clear skies and mild temperatures are forecast for the holiday weekend",
    "Hailstorms damaged crops across several farming districts last night",
    "The heat wave has pushed daytime temperatures above 40 degrees Celsius",
    "Snow accumulation of up to 30 centimeters is expected by Friday",
    "Tropical storm warnings have been issued for coastal communities",
    "Humidity levels above 80 percent make outdoor activities uncomfortable",
    "The jet stream is shifting southward bringing cooler air masses",
    "Lightning strikes caused multiple wildfires in the dry forest region",
]

if __name__ == "__main__":
    retriever = TinyRetriever()
    print("Indexing 50 documents...")
    retriever.index(corpus)
    print(f"Corpus shape: {retriever.corpus_embeddings.shape}\n")

    test_queries = [
        "How do I make a pasta dish?",
        "best practices for error handling in code",
        "will it rain tomorrow?",
        "tips for grilling meat",
        "how does version control work?",
    ]

    for q in test_queries:
        print(f"Query: {q}")
        results = retriever.query(q, k=3)
        for rank, (text, score) in enumerate(results, 1):
            print(f"  [{rank}] ({score:.3f}) {text}")
        print()

    opposite_queries = [
        "I love this product",
        "I hate this product",
    ]

    embs = retriever.model.encode(opposite_queries, normalize_embeddings=True)
    score = embs[0] @ embs[1]
    print(f"Cosine similarity between 'I love this' and 'I hate this': {score:.3f}")