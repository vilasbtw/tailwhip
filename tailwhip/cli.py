import typer
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

from tailwhip.ingestion.oracle_adapter import OracleAdapter
from tailwhip.ingestion.schema_normalizer import SchemaNormalizer
from tailwhip.indexing.document_builder import DocumentBuilder
from tailwhip.indexing.index_manager import IndexManager
from tailwhip.search.orchestrator import SearchOrchestrator


app     = typer.Typer(help="tailwhip — Oracle schema semantic search")
console = Console()


def _build_pipeline(base_dir: Path | None = None) -> SearchOrchestrator:
    index_manager = IndexManager(base_dir=base_dir)
    return SearchOrchestrator(index_manager=index_manager)


def _index_csv(file: Path, base_dir: Path | None = None) -> None:
    """Shared logic between `index` and `refresh` commands."""
    if not file.exists():
        console.print(f"[red]Arquivo não encontrado:[/red] {file}")
        raise typer.Exit(1)

    console.print(f"[dim]Lendo[/dim] {file} ...")

    adapter    = OracleAdapter()
    normalizer = SchemaNormalizer()
    builder    = DocumentBuilder()

    columns = adapter.parse(file)
    tables  = normalizer.normalize(columns)

    from collections import defaultdict
    cols_by_table: dict[str, list] = defaultdict(list)
    for col in columns:
        cols_by_table[col.table_name].append(col)

    pairs     = [(table, cols_by_table[table.table_name]) for table in tables]
    documents = builder.build_all(pairs)

    console.print(
        f"[dim]Indexando[/dim] {len(documents)} tabelas "
        f"([dim]{len(columns)} colunas[/dim]) ..."
    )

    index_manager = IndexManager(base_dir=base_dir)
    index_manager.build(documents)

    console.print(f"[green]✓[/green] {len(documents)} tabelas indexadas.")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@app.command()
def index(
    file: Path = typer.Option(..., "--file", "-f", help="Path to the Oracle-exported CSV file"),
):
    """Index the Oracle metadata CSV from scratch."""
    _index_csv(file)


@app.command()
def refresh(
    file: Path = typer.Option(..., "--file", "-f", help="Path to the new Oracle-exported CSV file"),
):
    """Re-index after a new Oracle export."""
    console.print("[yellow]Reindexando...[/yellow]")
    _index_csv(file)


@app.command()
def search(
    query: str         = typer.Argument(..., help="Term or phrase to search for"),
    owners: Optional[str] = typer.Option(None,  "--owners", "-o", help="Filter by owners (comma-separated). e.g. OWNER1,OWNER2"),
    top: int           = typer.Option(10,    "--top",    "-n", help="Number of results to return"),
    show_columns: bool = typer.Option(False, "--show-columns",  help="Display relevant columns for each table"),
):
    """Search for tables relevant to a query."""
    try:
        pipeline = _build_pipeline()
    except RuntimeError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    owners_list = [o.strip() for o in owners.split(",")] if owners else None

    with console.status("[dim]Buscando...[/dim]"):
        results = pipeline.search(query, owners=owners_list, top_n=top)

    if not results:
        console.print("[yellow]Nenhum resultado encontrado.[/yellow]")
        raise typer.Exit(0)

    # Header panel
    filter_info = f" [dim](owner: {owners})[/dim]" if owners else ""
    console.print(
        Panel(
            Text(f'"{query}"', style="bold white"),
            title="[bold cyan]tailwhip[/bold cyan]",
            subtitle=f"{len(results)} resultado(s){filter_info}",
            border_style="cyan",
            expand=False,
        )
    )
    console.print()

    for result in results:
        _print_result(result, show_columns)


def _print_result(result, show_columns: bool) -> None:
    table = result.table
    score_bar = _score_bar(result.final_score)

    # Build result header line
    header = Text()
    header.append(f"#{result.rank} ", style="bold cyan")
    header.append(f"{table.schema_name + '.' if table.schema_name else ''}", style="dim")
    header.append(table.table_name, style="bold white")
    header.append(f"  {score_bar} ", style="green")
    header.append(f"{result.final_score:.3f}", style="dim")

    if table.fk_in_count > 0:
        header.append(f"  [dim]↑{table.fk_in_count} refs[/dim]")

    console.print(header)

    if table.description:
        console.print(f"   [dim]{table.description}[/dim]")

    if show_columns and result.relevant_columns:
        col_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
        col_table.add_column(style="dim cyan",  no_wrap=True)
        col_table.add_column(style="dim",        no_wrap=True)
        col_table.add_column(style="dim yellow", no_wrap=True)

        for col in result.relevant_columns:
            constraint = ""
            if col.is_pk:
                constraint = "[PK]"
            elif col.is_fk and col.fk_ref_table:
                ref = f"{col.fk_ref_schema}.{col.fk_ref_table}" if col.fk_ref_schema else col.fk_ref_table
                constraint = f"→ {ref}"

            col_table.add_row(col.column_name, col.data_type, constraint)

        console.print(col_table)

    console.print()


def _score_bar(score: float, width: int = 8) -> str:
    filled = round(score * width)
    return "█" * filled + "░" * (width - filled)


if __name__ == "__main__":
    app()