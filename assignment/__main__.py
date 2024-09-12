import click
from typing import Optional
from .assignment3 import Config, document_indexer_pipeline, document_search_pipeline

@click.group()
@click.option('--chromadb-path', default=None, help='Path to ChromaDB')
@click.pass_context
def cli(ctx, chromadb_path: Optional[str]):
    ctx.ensure_object(dict)
    ctx.obj['config'] = Config(chromadb_path=chromadb_path) if chromadb_path else Config()

@cli.command()
@click.argument('datafile', type=click.Path(exists=True))
@click.pass_context
def index(ctx, datafile: str):
    """Index documents from the given datafile."""
    config = ctx.obj['config']
    config.data_path = datafile
    document_indexer_pipeline(config)
    click.echo(f"Indexed documents from {datafile}")

@cli.command()
@click.option('--num', default=1, help='Number of results to return')
@click.argument('query', nargs=-1, type=str)
@click.pass_context
def search(ctx, num: int, query: tuple):
    """Search for documents using the given query."""
    config = ctx.obj['config']
    query_text = ' '.join(query)
    results = document_search_pipeline(config, query_text, num)
    if results:
        for i, (doc_id, score, doc_body) in enumerate(results):
            click.echo(f"Result {i+1}:")
            click.echo(f"  Document ID: {doc_id}")
            click.echo(f"  Score: {score}")
            click.echo(f"  Document body: {doc_body}")
    else:
        click.echo("No results found.")

if __name__ == '__main__':
    cli()