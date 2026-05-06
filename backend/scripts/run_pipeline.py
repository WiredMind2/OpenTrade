"""
Orchestrator to run the full MVP pipeline step-by-step and log outputs/errors.

Usage examples (PowerShell):
  python scripts/run_pipeline.py
  python scripts/run_pipeline.py --steps apply_schema,ingest_prices,ingest_news --continue-on-error

The script calls the existing scripts in `scripts/` using the same Python interpreter.
It writes a timestamped log to `logs/pipeline.log` and separate per-step logs under `logs/steps/`.
"""
import argparse
import subprocess
import sys
import os

# Add project root to sys.path so scripts can use absolute imports from `backend.scripts`
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from datetime import datetime
import io
from contextlib import redirect_stdout, redirect_stderr
import traceback
from backend.scripts.script_logger import logger


DEFAULT_STEPS = [
    'apply_schema',
    'download_kaggle',
    'ingest_prices',
    'scan_csvs',
    'ingest_minute_prices',
    'ingest_news',
    'scrape_articles',
    'map_articles_to_tickers',
    'labeling',
]


def db_table_count(db_path: str, query: str):
    import sqlite3
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(query)
        r = cur.fetchone()
        conn.close()
        return r[0] if r else 0
    except Exception as e:
        logger.warning("Database query failed in db_table_count: %s", e)
        return 0


def precheck_apply_schema(args):
    # Consider schema applied only if a small set of core tables exist.
    # (A partially-initialized DB might have only one table.)
    q = """
    SELECT COUNT(name)
    FROM sqlite_master
    WHERE type='table' AND name IN ('tickers', 'price_daily', 'sentiment_predictions', 'backtest_runs')
    """
    return db_table_count(args.db, q) >= 4


def precheck_download_kaggle(args):
    # skip if csv_dir contains any CSV files
    for root, _, files in os.walk(args.csv_dir):
        for f in files:
            if f.lower().endswith('.csv'):
                return True
    return False


def precheck_ingest_prices(args):
    q = "SELECT COUNT(*) FROM price_daily"
    return db_table_count(args.db, q) > 0


def precheck_scan_csvs(args):
    q = "SELECT COUNT(*) FROM tickers"
    return db_table_count(args.db, q) > 0


def precheck_ingest_news(args):
    # consider news ingested only if there exists at least one article with a publication date
    # that is within the range of available price data. This avoids skipping ingestion when
    # the DB contains only recent articles newer than our price history.
    q = "SELECT COUNT(*) FROM articles WHERE date(canonical_timestamp) <= (SELECT MAX(date) FROM price_daily)"
    return db_table_count(args.db, q) > 0


def precheck_scrape_articles(args):
    # skip scraping if there are no short/empty/cropped articles
    q = "SELECT COUNT(*) FROM articles WHERE url IS NOT NULL AND (content IS NULL OR length(content) < 200 OR content LIKE '%[+%')"
    cnt = db_table_count(args.db, q)
    return cnt == 0


def precheck_map_articles_to_tickers(args):
    q = "SELECT COUNT(*) FROM article_ticker"
    return db_table_count(args.db, q) > 0


def precheck_labeling(args):
    # check if ground truth labels exist
    q = "SELECT COUNT(*) FROM sentiment_predictions WHERE model='label_groundtruth'"
    return db_table_count(args.db, q) > 0


def precheck_ingest_minute_prices(args):
    q = "SELECT COUNT(*) FROM price_minute"
    return db_table_count(args.db, q) > 0


def precheck_backtest_runner(args):
    q = "SELECT COUNT(*) FROM backtest_runs"
    return db_table_count(args.db, q) > 0


STEP_PRECHECK = {
    'apply_schema': precheck_apply_schema,
    'download_kaggle': precheck_download_kaggle,
    'ingest_prices': precheck_ingest_prices,
    'scan_csvs': precheck_scan_csvs,
    'ingest_minute_prices': precheck_ingest_minute_prices,
    'ingest_news': precheck_ingest_news,
    'scrape_articles': precheck_scrape_articles,
    'map_articles_to_tickers': precheck_map_articles_to_tickers,
    'labeling': precheck_labeling,
    'backtest_runner': precheck_backtest_runner,
}


def postcheck_labeling(args, step_log):
    # check DB for ground-truth labels and if zero, run label_debug and attach its output
    import sqlite3, subprocess
    try:
        conn = sqlite3.connect(args.db)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM sentiment_predictions WHERE model='label_groundtruth'")
        cnt = cur.fetchone()[0]
        conn.close()
    except Exception as e:
        logger.warning('Postcheck labeling DB query failed: %s', e)
        cnt = 0
    if cnt == 0:
        logger.warning('Labeling produced 0 ground-truth labels. Running label_debug to collect diagnostics.')
        debug_script_path = os.path.join(os.path.dirname(__file__), 'label_debug.py')
        if not os.path.exists(debug_script_path):
            logger.warning('label_debug.py script not found at %s. Skipping diagnostics.', debug_script_path)
            return
        dbg_cmd = [sys.executable, debug_script_path, '--db', args.db, '--limit', str(args.debug_limit)]
        try:
            dbg = subprocess.run(dbg_cmd, capture_output=True, text=True)
            dbg_log_dir = os.path.join('logs', 'steps')
            os.makedirs(dbg_log_dir, exist_ok=True)
            dbg_log = os.path.join(dbg_log_dir, 'label_debug_from_pipeline.log')
            with open(dbg_log, 'w', encoding='utf-8') as f:
                f.write('COMMAND: ' + ' '.join(dbg_cmd) + '\n\n')
                f.write('=== STDOUT ===\n')
                f.write(dbg.stdout or '')
                f.write('\n=== STDERR ===\n')
                f.write(dbg.stderr or '')
            logger.warning('Wrote label debug output to %s', dbg_log)
        except Exception as e:
            logger.exception('Failed to run label_debug: %s', e)


STEP_POSTCHECK = {
    'labeling': postcheck_labeling,
}




def run_step(name: str, func, args, log_dir: str):
    step_log_dir = os.path.join(log_dir, 'steps')
    os.makedirs(step_log_dir, exist_ok=True)
    step_log = os.path.join(step_log_dir, f'{name}.log')
    logger.info('Running step %s', name)
    try:
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            func(args)
        stdout = stdout_capture.getvalue()
        stderr = stderr_capture.getvalue()
        returncode = 0
    except Exception as e:
        # Preserve any partial output we captured before the exception,
        # and include the full traceback in the step log for debugging.
        try:
            stdout = stdout_capture.getvalue()  # type: ignore[name-defined]
            stderr = stderr_capture.getvalue()  # type: ignore[name-defined]
        except Exception:
            stdout = ""
            stderr = ""
        tb = traceback.format_exc()
        stderr = (stderr + ("\n" if stderr else "") + tb).strip()
        returncode = 1
        logger.exception("Step %s raised an exception", name)
    # write stdout/stderr to step log
    with open(step_log, 'w', encoding='utf-8') as f:
        f.write('COMMAND: run_' + name + '\n')
        f.write('ARGS: ' + str(vars(args)) + '\n\n')
        f.write('=== STDOUT ===\n')
        f.write(stdout or '')
        f.write('\n=== STDERR ===\n')
        f.write(stderr or '')
    if returncode == 0:
        logger.info('Step %s completed successfully', name)
        # summarize step log (last few lines) into main log for visibility
        try:
            lines = (stdout + stderr).splitlines()
            tail = '\n'.join(lines[-10:]) if lines else ''
            logger.info('Step %s log tail:\n%s', name, tail)
        except Exception as e:
            logger.warning("Failed to read log tail for step %s: %s", name, e)
        return True, step_log
    else:
        logger.error('Step %s failed with exception', name)
        try:
            lines = (stdout + stderr).splitlines()
            tail = '\n'.join(lines[-40:]) if lines else ''
            logger.error('Step %s log tail:\n%s', name, tail)
        except Exception:
            pass
        return False, step_log


def run_apply_schema(args):
    from backend.scripts import apply_schema
    schema_path = os.path.join(os.path.dirname(args.db), '..', 'db', 'schema.sql')
    apply_schema.apply_schema(args.db, schema_path)


def run_download_kaggle(args):
    from backend.scripts import download_kaggle
    if download_kaggle.KAGGLE_USERNAME and download_kaggle.KAGGLE_KEY:
        download_kaggle.write_kaggle_json(download_kaggle.KAGGLE_USERNAME, download_kaggle.KAGGLE_KEY)
    else:
        print('KAGGLE_USERNAME and KAGGLE_KEY not set in environment. If you have a kaggle.json in ~/.kaggle, the script will use it. Otherwise, set credentials in .env or download manually from the Kaggle web UI.')
    download_kaggle.download_dataset(args.kaggle_dataset, args.csv_dir)


def run_ingest_prices(args):
    import os
    from backend.scripts import ingest_prices
    # recursively find CSVs
    paths = []
    for root, _, files in os.walk(args.csv_dir):
        for f in files:
            if f.lower().endswith('.csv'):
                paths.append(os.path.join(root, f))
    if not paths:
        print('No CSV files found under', args.csv_dir)
        return
    for p in paths:
        ingest_prices.ingest_csv_to_db(args.db, p, None)


def run_scan_csvs(args):
    from backend.scripts import scan_csvs
    scan_csvs.scan_and_register(args.db, args.csv_dir)


def run_ingest_news(args):
    from backend.scripts import ingest_news
    if not ingest_news.NEWSAPI_KEY:
        print('NEWSAPI_KEY not set in environment. Export it or add to .env file.')
        return
    conn = ingest_news.NewsAPIConnector(api_key=ingest_news.NEWSAPI_KEY)
    articles = conn.fetch_headlines(query=args.news_query, from_dt=getattr(args, 'news_from', None), to_dt=getattr(args, 'news_to', None))
    ingest_news.store_articles(args.db, articles)


def run_scrape_articles(args):
    from backend.scripts import scrape_articles
    scrape_articles.scrape(args.db, args.scrape_limit, args.scrape_pause)


def run_map_articles_to_tickers(args):
    import sqlite3
    from backend.scripts import map_articles_to_tickers
    conn = sqlite3.connect(args.db)
    tickers = map_articles_to_tickers.load_tickers(conn)
    if not tickers:
        print('No tickers found in DB. Run scripts/scan_csvs.py first to register tickers from your CSVs.')
    else:
        print(f'Loaded {len(tickers)} tickers from the DB')
        mapped = map_articles_to_tickers.map_articles(conn, tickers)
        print(f'Inserted {mapped} article->ticker mappings')
    conn.close()


def run_labeling(args):
    from backend.scripts import labeling
    labeling.label_articles(args.db, args.horizons)


def run_ingest_minute_prices(args):
    from backend.scripts import ingest_minute_prices
    # Fetch minute data for top tickers (you can modify this list)
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    tickers_str = ','.join(tickers)
    # Run the script with default parameters
    import subprocess
    import sys
    result = subprocess.run([
        sys.executable, 'backend/scripts/ingest_minute_prices.py',
        '--tickers', tickers_str,
        '--period', '30d',
        '--interval', '1m'
    ], capture_output=True, text=True)
    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        raise Exception(f"ingest_minute_prices failed with code {result.returncode}")


def run_backtest_runner(args):
    from backend.scripts import backtest_runner
    backtest_runner.run_backtest(args.db, args.backtest_start, args.backtest_end)


STEP_COMMANDS = {
    'apply_schema': run_apply_schema,
    'download_kaggle': run_download_kaggle,
    'ingest_prices': run_ingest_prices,
    'scan_csvs': run_scan_csvs,
    'ingest_minute_prices': run_ingest_minute_prices,
    'ingest_news': run_ingest_news,
    'scrape_articles': run_scrape_articles,
    'map_articles_to_tickers': run_map_articles_to_tickers,
    'labeling': run_labeling,
    'backtest_runner': run_backtest_runner,
}


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--db', default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'backtest.db')))
    parser.add_argument('--csv_dir', default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'kaggle_yahoo')))
    parser.add_argument('--kaggle_dataset', default='iveeaten3223times/massive-yahoo-finance-dataset')
    parser.add_argument('--news_query', default='earnings OR acquisition OR merger OR revenue')
    parser.add_argument('--news_from', default=None, help='Optional YYYY-MM-DD start date for news ingestion')
    parser.add_argument('--news_to', default=None, help='Optional YYYY-MM-DD end date for news ingestion')
    parser.add_argument('--horizons', nargs='+', type=int, default=[1, 3, 7])
    parser.add_argument('--scrape_limit', type=int, default=200)
    parser.add_argument('--scrape_pause', type=float, default=1.0)
    parser.add_argument('--backtest_start', default='2020-01-01')
    parser.add_argument('--backtest_end', default='2025-01-01')
    parser.add_argument('--steps', default=','.join(DEFAULT_STEPS))
    parser.add_argument('--continue-on-error', action='store_true')
    parser.add_argument('--no-precheck', action='store_true', help='Run requested steps even if pre-checks indicate they are completed')
    parser.add_argument('--debug_limit', type=int, default=20)
    args = parser.parse_args()

    log_dir = 'logs'

    requested = [s.strip() for s in args.steps.split(',') if s.strip()]
    logger.info('Pipeline started. Steps: %s', requested)
    start_time = datetime.utcnow()
    for step in requested:
        if step not in STEP_COMMANDS:
            logger.warning('Unknown step "%s" - skipping', step)
            continue

        # pre-check: skip steps which appear already completed
        if not args.no_precheck:
            pre = STEP_PRECHECK.get(step)
            try:
                if pre and pre(args):
                    logger.info('Skipping step %s because pre-check indicates it is already completed', step)
                    continue
            except Exception as e:
                logger.warning('Pre-check for step %s raised an exception: %s (will attempt to run step)', step, e)

        func = STEP_COMMANDS[step]
        ok, step_log = run_step(step, func, args, log_dir)
        if not ok:
            if not args.continue_on_error:
                logger.error('Aborting pipeline due to failure in step %s', step)
                break
            else:
                logger.warning('Continuing pipeline despite failure in step %s', step)
        else:
            # run any post-checks for this step
            post = STEP_POSTCHECK.get(step)
            try:
                if post:
                    post(args, step_log)
            except Exception as e:
                logger.exception('Post-check for step %s raised exception: %s', step, e)
    end_time = datetime.utcnow()
    logger.info('Pipeline finished. Duration: %s', end_time - start_time)


if __name__ == '__main__':
    main()
