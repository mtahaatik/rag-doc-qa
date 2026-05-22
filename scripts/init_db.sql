-- Enable the pgvector extension. Must run as superuser, which is the default
-- in the docker image's init phase.
CREATE EXTENSION IF NOT EXISTS vector;

-- Documents we have ingested. Source is whatever the user calls it
-- (a filename, a URL, a notes label). Kept simple on purpose.
CREATE TABLE IF NOT EXISTS documents (
    id          BIGSERIAL PRIMARY KEY,
    source      TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Each document is split into chunks. Each chunk has its own embedding.
-- The embedding dimension (384) must match the EMBEDDING_DIM env var and
-- the model output. all-MiniLM-L6-v2 produces 384-dim vectors.
CREATE TABLE IF NOT EXISTS chunks (
    id           BIGSERIAL PRIMARY KEY,
    document_id  BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index  INT NOT NULL,
    content      TEXT NOT NULL,
    embedding    vector(384) NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Approximate nearest neighbor index. ivfflat is the simpler option; hnsw
-- gives better recall but takes longer to build. For a learning project
-- ivfflat with a small number of lists is fine.
-- Note: this index should be created AFTER you have some data, otherwise
-- the lists parameter is meaningless. Left here so the schema is complete.
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

CREATE INDEX IF NOT EXISTS chunks_document_id_idx ON chunks(document_id);
