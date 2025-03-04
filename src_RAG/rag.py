
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class Rag:
    def __init__(self, file_path, model_name='all-MiniLM-L6-v2', max_length=500):
        """
        Inicializa el indexador cargando el texto, dividiéndolo en fragmentos y calculando sus embeddings.

        Args:
            file_path (str): Ruta del archivo de texto.
            model_name (str, opcional): Nombre del modelo de embeddings. Por defecto 'all-MiniLM-L6-v2'.
            max_length (int, opcional): Número máximo de palabras por fragmento. Por defecto 500.
        """
        self.file_path = file_path
        self.max_length = max_length
        self.model = SentenceTransformer(model_name)
        self.text = self.load_text()
        self.fragments = self.split_text(self.text)
        self.embeddings = self.make_embeddings(self.fragments)

    def load_text(self):
        """Carga el contenido del archivo de texto."""
        with open(self.file_path, 'r', encoding='utf-8') as file:
            return file.read()

    def split_text(self, text):
        """
        Divide el texto en fragmentos de aproximadamente 'max_length' palabras.

        Args:
            text (str): Texto completo a dividir.

        Returns:
            List[str]: Lista de fragmentos.
        """
        words = text.split()
        return [' '.join(words[i:i + self.max_length]) for i in range(0, len(words), self.max_length)]

    def make_embeddings(self, fragments):
        """
        Calcula los embeddings para cada fragmento.

        Args:
            fragments (List[str]): Lista de fragmentos de texto.

        Returns:
            np.array: Array con los embeddings correspondientes.
        """
        return self.model.encode(fragments)

    def get_chunk_relevates(self, query, top_k=3):
        """
        Devuelve los fragmentos más relevantes para una consulta dada.

        Args:
            query (str): Consulta a evaluar.
            top_k (int, opcional): Número de fragmentos a devolver. Por defecto 3.

        Returns:
            List[str]: Lista de fragmentos relevantes.
        """
        query_emb = self.model.encode([query])
        similarities = cosine_similarity(query_emb, self.embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:]
        return [self.fragments[i] for i in top_indices]
