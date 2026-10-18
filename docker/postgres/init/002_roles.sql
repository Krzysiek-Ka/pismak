-- Role bazy RAG. Superuser z POSTGRES_USER służy wyłącznie do administracji:
--   pismak_ingest — ingestion dokumentów; właściciel tabel rag_documents i rag_chunks,
--   pismak_app    — interfejs i wyszukiwanie; tylko SELECT na tych tabelach.
--
-- Przy pierwszym starcie kontenera skrypt uruchamia obraz PostgreSQL. Na bazie
-- utworzonej wcześniej wykonaj go ręcznie (README, punkt 4.2); można go powtarzać.
-- Hasła pochodzą ze zmiennych środowiskowych kontenera (docker-compose.yaml).

\set ON_ERROR_STOP on
\getenv ingest_password PISMAK_INGEST_DB_PASSWORD
\getenv app_password PISMAK_APP_DB_PASSWORD

\if :{?ingest_password}
\else
    DO $$ BEGIN RAISE EXCEPTION 'Brak PISMAK_INGEST_DB_PASSWORD w środowisku kontenera.'; END $$;
\endif
\if :{?app_password}
\else
    DO $$ BEGIN RAISE EXCEPTION 'Brak PISMAK_APP_DB_PASSWORD w środowisku kontenera.'; END $$;
\endif

SELECT 'CREATE ROLE pismak_ingest'
WHERE NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'pismak_ingest') \gexec
ALTER ROLE pismak_ingest WITH LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION
    PASSWORD :'ingest_password';

SELECT 'CREATE ROLE pismak_app'
WHERE NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'pismak_app') \gexec
ALTER ROLE pismak_app WITH LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION
    PASSWORD :'app_password';
-- Druga linia obrony: nawet przypadkowo nadane prawo zapisu nie zadziała w sesji aplikacji.
ALTER ROLE pismak_app SET default_transaction_read_only = on;

-- Obiekty w schemacie public tworzy tylko rola ingestion (w PostgreSQL 15+ to ustawienie
-- domyślne; tu jest jawne).
REVOKE CREATE ON SCHEMA public FROM PUBLIC;
GRANT CONNECT ON DATABASE :"DBNAME" TO pismak_ingest, pismak_app;
GRANT USAGE, CREATE ON SCHEMA public TO pismak_ingest;
GRANT USAGE ON SCHEMA public TO pismak_app;

-- Baza sprzed rozdzielenia ról: tabele należą do superusera. Ingestion zmienia schemat
-- (ALTER TABLE, CREATE INDEX), więc musi być ich właścicielem.
SELECT format('ALTER TABLE public.%I OWNER TO pismak_ingest', tablename)
FROM pg_tables
WHERE schemaname = 'public' AND tablename IN ('rag_documents', 'rag_chunks') \gexec

SELECT format('GRANT SELECT ON TABLE public.%I TO pismak_app', tablename)
FROM pg_tables
WHERE schemaname = 'public' AND tablename IN ('rag_documents', 'rag_chunks') \gexec

-- Tabele, które ingestion utworzy później (świeża baza), od razu są czytelne dla aplikacji.
ALTER DEFAULT PRIVILEGES FOR ROLE pismak_ingest IN SCHEMA public
    GRANT SELECT ON TABLES TO pismak_app;
