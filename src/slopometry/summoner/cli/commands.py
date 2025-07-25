"""CLI commands for summoner features."""

import sys
from pathlib import Path

import click
from rich.console import Console

from slopometry.display.formatters import (
    create_experiment_table,
    create_features_table,
    create_nfp_objectives_table,
    create_progress_history_table,
    create_user_story_entries_table,
)
from slopometry.summoner.services.experiment_service import ExperimentService
from slopometry.summoner.services.llm_service import LLMService
from slopometry.summoner.services.nfp_service import NFPService
from slopometry.summoner.services.user_story_service import UserStoryService

console = Console()


def complete_experiment_id(ctx, param, incomplete):
    """Complete experiment IDs from the database."""
    try:
        experiment_service = ExperimentService()
        experiments = experiment_service.list_experiments()
        return [exp["id"] for exp in experiments if exp["id"].startswith(incomplete)]
    except Exception:
        return []


def complete_nfp_id(ctx, param, incomplete):
    """Complete NFP objective IDs from the database."""
    try:
        nfp_service = NFPService()
        objectives = nfp_service.list_nfp_objectives()
        return [obj.id for obj in objectives if obj.id.startswith(incomplete)]
    except Exception:
        return []


def complete_feature_id(ctx, param, incomplete):
    """Complete feature IDs from the database."""
    try:
        from pathlib import Path

        from slopometry.core.database import EventDatabase

        db = EventDatabase()
        repo_path = Path.cwd()
        feature_ids = db.get_feature_ids_for_completion(repo_path)
        return [fid for fid in feature_ids if fid.startswith(incomplete)]
    except Exception:
        return []


def complete_user_story_entry_id(ctx, param, incomplete):
    """Complete user story entry IDs from the database."""
    try:
        from slopometry.core.database import EventDatabase

        db = EventDatabase()
        entry_ids = db.get_user_story_entry_ids_for_completion()
        return [eid for eid in entry_ids if eid.startswith(incomplete)]
    except Exception:
        return []


@click.group()
def summoner():
    """Summoner commands for advanced experimentation and AI integration."""
    pass


@summoner.command("run-experiments")
@click.option("--commits", "-c", default=5, help="Number of commits to analyze (default: 5)")
@click.option("--max-workers", "-w", default=4, help="Maximum parallel workers (default: 4)")
@click.option(
    "--repo-path",
    "-r",
    type=click.Path(exists=True, path_type=Path),
    help="Repository path (default: current directory)",
)
def run_experiments(commits: int, max_workers: int, repo_path: Path | None):
    """Run parallel experiments across git commits to track and analyze code complexity evolution patterns."""
    if repo_path is None:
        repo_path = Path.cwd()

    experiment_service = ExperimentService()

    console.print(f"[bold]Running {commits} experiments with up to {max_workers} workers[/bold]")
    console.print(f"Repository: {repo_path}")

    try:
        experiments = experiment_service.run_parallel_experiments(repo_path, commits, max_workers)

        console.print(f"\\n[green]✓ Completed {len(experiments)} experiments[/green]")

        for experiment in experiments.values():
            status_color = "green" if experiment.status.value == "completed" else "red"
            console.print(
                f"  {experiment.start_commit} → {experiment.target_commit}: [{status_color}]{experiment.status.value}[/]"
            )

    except Exception as e:
        console.print(f"[red]Failed to run experiments: {e}[/red]")
        sys.exit(1)


@summoner.command("analyze-commits")
@click.option("--base-commit", "-b", default="HEAD~10", help="Base commit (default: HEAD~10)")
@click.option("--head-commit", "-h", default="HEAD", help="Head commit (default: HEAD)")
@click.option(
    "--repo-path",
    "-r",
    type=click.Path(exists=True, path_type=Path),
    help="Repository path (default: current directory)",
)
def analyze_commits(base_commit: str, head_commit: str, repo_path: Path | None):
    """Analyze complexity evolution across a chain of commits."""
    if repo_path is None:
        repo_path = Path.cwd()

    experiment_service = ExperimentService()

    console.print(f"[bold]Analyzing commits from {base_commit} to {head_commit}[/bold]")
    console.print(f"Repository: {repo_path}")

    try:
        experiment_service.analyze_commit_chain(repo_path, base_commit, head_commit)
        console.print("\\n[green]✓ Analysis complete[/green]")

    except Exception as e:
        console.print(f"[red]Failed to analyze commits: {e}[/red]")
        sys.exit(1)


@summoner.command("list-experiments")
def list_experiments():
    """List all experiment runs."""
    experiment_service = ExperimentService()
    experiments_data = experiment_service.list_experiments()

    if not experiments_data:
        console.print("[yellow]No experiments found[/yellow]")
        return

    table = create_experiment_table(experiments_data)
    console.print(table)


@summoner.command("show-experiment")
@click.argument("experiment_id", shell_complete=complete_experiment_id)
def show_experiment(experiment_id: str):
    """Show detailed progress for an experiment."""
    experiment_service = ExperimentService()

    result = experiment_service.get_experiment_details(experiment_id)
    if not result:
        console.print(f"[red]Experiment {experiment_id} not found[/red]")
        return

    experiment_row, progress_rows = result

    console.print(f"[bold]Experiment: {experiment_row[0]}[/bold]")
    console.print(f"Repository: {experiment_row[1]}")
    console.print(f"Commits: {experiment_row[2]} → {experiment_row[3]}")
    console.print(f"Status: {experiment_row[8]}")

    if progress_rows:
        progress_data = experiment_service.prepare_progress_data_for_display(progress_rows)
        table = create_progress_history_table(progress_data)
        console.print(table)

        # Show final score
        final_row = progress_rows[-1]
        final_cli = final_row[1]
        console.print(f"\\n[bold]Final CLI Score: {final_cli:.3f}[/bold]")
    else:
        console.print("[yellow]No progress data found[/yellow]")


@summoner.command("userstorify")
@click.option("--base-commit", "-b", help="Base commit (required with --head-commit)")
@click.option("--head-commit", "-h", help="Head commit (required with --base-commit)")
@click.option(
    "--feature-id",
    "-f",
    shell_complete=complete_feature_id,
    help="Feature ID to analyze (alternative to --base-commit/--head-commit)",
)
@click.option(
    "--repo-path",
    "-r",
    type=click.Path(exists=True, path_type=Path),
    help="Repository path (default: current directory)",
)
def userstorify(base_commit: str | None, head_commit: str | None, feature_id: str | None, repo_path: Path | None):
    """Generate user stories from commits using configured AI agents and save permanently to user story collection."""
    if repo_path is None:
        repo_path = Path.cwd()

    llm_service = LLMService()

    # Handle feature ID resolution
    if feature_id:
        if base_commit or head_commit:
            console.print("[red]Cannot specify both --feature-id and --base-commit/--head-commit[/red]")
            sys.exit(1)

        from slopometry.core.database import EventDatabase

        db = EventDatabase()

        # Use match statement with length-based strategy
        match len(feature_id):
            case 36:  # Full UUID4 length
                feature = db.get_feature_boundary_by_id(feature_id, repo_path)
                if not feature:
                    console.print(f"[red]Feature ID not found: {feature_id}[/red]")
                    console.print("[dim]Use 'slopometry summoner list-features' to see all available features[/dim]")
                    sys.exit(1)
            case length if length < 36:  # Partial ID
                full_id = db.resolve_feature_id(feature_id, repo_path)
                if full_id is None:
                    console.print(f"[red]Ambiguous or not found feature ID: {feature_id}[/red]")
                    console.print("[dim]Use 'slopometry summoner list-features' to see all available features[/dim]")
                    sys.exit(1)
                feature = db.get_feature_boundary_by_id(full_id, repo_path)
            case _:  # Invalid length
                console.print(f"[red]Invalid feature ID format: {feature_id}[/red]")
                console.print("[dim]Feature IDs should be UUID4 format (36 chars) or partial (< 36 chars)[/dim]")
                sys.exit(1)

        base_commit = feature.base_commit
        head_commit = feature.head_commit

        console.print(f"[bold]Generating user stories for feature: {feature.short_id}[/bold]")
        console.print(f"Feature: {feature.feature_message}")
        console.print(f"Commits: {base_commit[:8]} → {head_commit[:8]}")
    elif base_commit and head_commit:
        console.print(f"[bold]Generating user stories from {base_commit} to {head_commit}[/bold]")
    else:
        console.print("[red]Must specify either --feature-id or both --base-commit and --head-commit[/red]")
        console.print("[dim]Use 'slopometry summoner list-features' to see available features[/dim]")
        sys.exit(1)

    console.print(f"Repository: {repo_path}")
    console.print(f"Using agents: {', '.join(llm_service.get_configured_agents())}")

    # Get commit info for display
    commit_info = llm_service.get_commit_info_for_display(base_commit, head_commit)

    console.print("\\n[yellow]Resolving commit references...[/yellow]")
    console.print(f"Base: {commit_info['base_display']}")
    console.print(f"Head: {commit_info['head_display']}")
    console.print(f"Stride size: {commit_info['stride_size']} commits")

    console.print("\\n[yellow]Fetching commit diff...[/yellow]")

    try:
        successful_generations, error_messages = llm_service.generate_user_stories_from_commits(
            repo_path, base_commit, head_commit
        )

        for error in error_messages:
            console.print(f"[yellow]Warning: {error}[/yellow]")

        if successful_generations > 0:
            console.print(
                f"\\n[bold green]Successfully generated {successful_generations} user story entries[/bold green]"
            )
            console.print("[dim]View with: slopometry summoner list-user-stories[/dim]")
        else:
            console.print("\\n[red]Failed to generate any user stories[/red]")

    except Exception as e:
        console.print(f"[red]Failed to generate user stories: {e}[/red]")
        sys.exit(1)


@summoner.command("rate-user-stories")
@click.option("--limit", "-l", default=10, help="Number of entries to show for rating (default: 10)")
@click.option("--filter-model", help="Filter by specific model")
@click.option("--unrated-only", is_flag=True, help="Show only unrated entries (rating = 3)")
def rate_user_stories(limit: int, filter_model: str | None, unrated_only: bool):
    """Rate existing user stories in the user story collection."""
    user_story_service = UserStoryService()

    try:
        filtered_entries = user_story_service.filter_entries_for_rating(limit, filter_model, unrated_only)

        if not filtered_entries:
            console.print("[yellow]No entries match the specified filters[/yellow]")
            return

        console.print(f"[bold]Rating {len(filtered_entries)} user story entries[/bold]")
        if filter_model:
            console.print(f"[dim]Filtered by model: {filter_model}[/dim]")
        if unrated_only:
            console.print("[dim]Showing only unrated entries[/dim]")

        updated_count = 0

        for i, entry in enumerate(filtered_entries, 1):
            console.print(f"\\n[bold cyan]Entry {i}/{len(filtered_entries)}[/bold cyan]")
            console.print(f"[dim]ID: {entry.id}[/dim]")
            console.print(f"[dim]Created: {entry.created_at.strftime('%Y-%m-%d %H:%M')}[/dim]")
            console.print(f"[dim]Model: {entry.model_used}[/dim]")
            console.print(
                f"[dim]Commits: {entry.base_commit[:8]}→{entry.head_commit[:8]} (stride: {entry.stride_size})[/dim]"
            )
            console.print(f"[dim]Current rating: {entry.rating}/5[/dim]")

            console.print("\\n[bold]Diff Preview:[/bold]")
            diff_preview = entry.diff_content[:500]
            if len(entry.diff_content) > 500:
                diff_preview += "... [truncated]"
            console.print(f"[dim]{diff_preview}[/dim]")

            console.print("\\n[bold]Generated User Stories:[/bold]")
            stories_preview = entry.user_stories[:1000]
            if len(entry.user_stories) > 1000:
                stories_preview += "... [truncated]"
            console.print(stories_preview)

            try:
                new_rating = click.prompt(
                    f"\\nRate this user story generation (1-5, current: {entry.rating}, 's' to skip, 'q' to quit)",
                    type=str,
                )

                if new_rating.lower() == "q":
                    break
                elif new_rating.lower() == "s":
                    console.print("[yellow]Skipped[/yellow]")
                    continue

                new_rating_int = int(new_rating)
                if not (1 <= new_rating_int <= 5):
                    console.print("[red]Rating must be between 1 and 5[/red]")
                    continue

                guidelines = click.prompt(
                    "Guidelines for improving (optional)", default=entry.guidelines_for_improving, show_default=False
                )

                user_story_service.rate_user_story_entry(entry, new_rating_int, guidelines)
                updated_count += 1

                console.print(f"[green]✓ Updated rating to {new_rating_int}/5[/green]")

            except (ValueError, click.Abort):
                console.print("[yellow]Invalid input, skipping[/yellow]")
                continue

        console.print(f"\\n[bold green]Updated {updated_count} entries[/bold green]")

    except Exception as e:
        console.print(f"[red]Failed to rate user stories: {e}[/red]")


@summoner.command("list-features")
@click.option(
    "--limit",
    "-l",
    default=20,
    help="Maximum number of feature merges to analyze (default: 20)",
)
@click.option(
    "--repo-path",
    "-r",
    type=click.Path(exists=True, path_type=Path),
    help="Repository path (default: current directory)",
)
def list_features(limit: int, repo_path: Path | None):
    """List detected feature boundaries from merge commits."""
    if repo_path is None:
        repo_path = Path.cwd()

    console.print(f"[bold]Detecting feature boundaries in {repo_path}[/bold]")

    llm_service = LLMService()

    try:
        features = llm_service.get_feature_boundaries(repo_path, limit)

        if not features:
            console.print("[yellow]No feature merge commits found[/yellow]")
            return

        features_data = llm_service.prepare_features_data_for_display(features)
        table = create_features_table(features_data)
        console.print(table)

        console.print("\\n[dim]To analyze a feature, use:[/dim]")
        console.print("[cyan]slopometry summoner userstorify --feature-id <feature-id>[/cyan]")
        console.print("[dim]Or with specific commits:[/dim]")
        console.print("[cyan]slopometry summoner userstorify --base-commit <base> --head-commit <head>[/cyan]")

    except Exception as e:
        console.print(f"[red]Failed to list features: {e}[/red]")
        sys.exit(1)


@summoner.command("user-story-stats")
def user_story_stats():
    """Show statistics about the collected user stories."""
    user_story_service = UserStoryService()

    try:
        stats = user_story_service.get_user_story_statistics()

        console.print("[bold]User Story Statistics[/bold]\\n")
        console.print(f"Total entries: {stats.total_entries}")
        console.print(f"Average rating: {stats.avg_rating}/5")
        console.print(f"Unique models used: {stats.unique_models}")
        console.print(f"Unique repositories: {stats.unique_repos}")

        if stats.rating_distribution:
            console.print("\\n[bold]Rating Distribution:[/bold]")
            for rating, count in stats.rating_distribution.items():
                console.print(f"  {rating}/5: {count} entries")

        if stats.total_entries > 0:
            console.print("\\n[dim]Use 'slopometry summoner list-user-stories' to view individual entries[/dim]")

    except Exception as e:
        console.print(f"[red]Failed to get user story statistics: {e}[/red]")


@summoner.command("list-user-stories")
@click.option("--limit", "-l", default=10, help="Number of entries to show (default: 10)")
def list_user_stories(limit: int):
    """Show recent user story entries."""
    user_story_service = UserStoryService()

    try:
        entries = user_story_service.get_user_story_entries(limit)

        if not entries:
            console.print("[yellow]No user story entries found[/yellow]")
            return

        entries_data = user_story_service.prepare_entries_data_for_display(entries)
        table = create_user_story_entries_table(entries_data, len(entries))
        console.print(table)

    except Exception as e:
        console.print(f"[red]Failed to get user story entries: {e}[/red]")


@summoner.command("user-story-export")
@click.option("--output", "-o", type=click.Path(), help="Output file path (default: slopometry_user_stories.parquet)")
@click.option("--upload-to-hf", is_flag=True, help="Upload to Hugging Face after export")
@click.option("--hf-repo", help="Hugging Face user story repository (e.g., username/repository-name)")
def user_story_export(output: str | None, upload_to_hf: bool, hf_repo: str | None):
    """Export user stories to Parquet format."""
    user_story_service = UserStoryService()

    if output is None:
        output = "slopometry_user_stories.parquet"
    output_path = Path(output)

    try:
        console.print(f"[yellow]Exporting user stories to {output_path}...[/yellow]")

        try:
            count = user_story_service.export_user_stories(output_path)
        except ImportError as e:
            console.print(f"[red]Missing required dependencies for export: {e}[/red]")
            console.print("Install with: pip install pandas pyarrow")
            return

        if count == 0:
            console.print("[yellow]No user story entries to export[/yellow]")
            return

        console.print(f"[green]✓ Exported {count} entries to {output_path}[/green]")

        if upload_to_hf:
            try:
                hf_repo_name = user_story_service.upload_to_huggingface(output_path, hf_repo)
                console.print(f"\\n[yellow]Uploading to Hugging Face: {hf_repo_name}...[/yellow]")
                console.print(
                    f"[green]✓ Successfully uploaded to https://huggingface.co/datasets/{hf_repo_name}[/green]"
                )
            except ImportError as e:
                console.print(f"[red]{e}[/red]")
            except Exception as e:
                console.print(f"[red]Failed to upload to Hugging Face: {e}[/red]")

    except Exception as e:
        console.print(f"[red]Failed to export user stories: {e}[/red]")


@summoner.command("show-user-story")
@click.argument("entry_id", shell_complete=complete_user_story_entry_id)
def show_user_story(entry_id: str):
    """Show detailed information for a user story entry."""
    from slopometry.core.database import EventDatabase

    db = EventDatabase()

    # Use match statement with length-based strategy like feature IDs
    match len(entry_id):
        case 36:  # Full UUID4 length
            entry = db.get_user_story_entry_by_id(entry_id)
            if not entry:
                console.print(f"[red]User story entry not found: {entry_id}[/red]")
                console.print("[dim]Use 'slopometry summoner list-user-stories' to see all entries[/dim]")
                sys.exit(1)
        case length if length < 36:  # Partial ID
            full_id = db.resolve_user_story_entry_id(entry_id)
            if full_id is None:
                console.print(f"[red]Ambiguous or not found user story entry ID: {entry_id}[/red]")
                console.print("[dim]Use 'slopometry summoner list-user-stories' to see all entries[/dim]")
                sys.exit(1)
            entry = db.get_user_story_entry_by_id(full_id)
        case _:  # Invalid length
            console.print(f"[red]Invalid user story entry ID format: {entry_id}[/red]")
            console.print("[dim]Entry IDs should be UUID4 format (36 chars) or partial (< 36 chars)[/dim]")
            sys.exit(1)

    # Display entry details
    console.print(f"[bold]User Story Entry: {entry.short_id}[/bold]")
    console.print(f"Created: {entry.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
    console.print(f"Repository: {entry.repository_path}")
    console.print(f"Model: {entry.model_used}")
    console.print(f"Rating: {entry.rating}/5")
    console.print(f"Commits: {entry.base_commit[:8]} → {entry.head_commit[:8]} (stride: {entry.stride_size})")

    if entry.guidelines_for_improving:
        console.print("\\n[bold]Guidelines for improving:[/bold]")
        console.print(entry.guidelines_for_improving)

    console.print("\\n[bold]Diff Content:[/bold]")
    diff_preview = entry.diff_content[:1000]
    if len(entry.diff_content) > 1000:
        diff_preview += "\\n\\n... [truncated, showing first 1000 chars]"
    console.print(f"[dim]{diff_preview}[/dim]")

    console.print("\\n[bold]Generated User Stories:[/bold]")
    console.print(entry.user_stories)


@summoner.command("list-nfp")
@click.option("--repo-path", "-r", type=click.Path(exists=True, path_type=Path), help="Repository path filter")
def list_nfp(repo_path: Path | None):
    """List all NFP objectives."""
    nfp_service = NFPService()

    try:
        repo_filter = str(repo_path) if repo_path else None
        objectives = nfp_service.list_nfp_objectives(repo_filter)

        if not objectives:
            console.print("[yellow]No NFP objectives found[/yellow]")
            return

        objectives_data = nfp_service.prepare_objectives_data_for_display(objectives)
        table = create_nfp_objectives_table(objectives_data)
        console.print(table)

    except Exception as e:
        console.print(f"[red]Failed to list NFP objectives: {e}[/red]")


@summoner.command("show-nfp")
@click.argument("nfp_id", shell_complete=complete_nfp_id)
def show_nfp(nfp_id: str):
    """Show detailed information for an NFP objective."""
    nfp_service = NFPService()

    try:
        nfp = nfp_service.get_nfp_objective(nfp_id)
        if not nfp:
            console.print(f"[red]NFP objective {nfp_id} not found[/red]")
            return

        summary = nfp_service.get_objective_summary(nfp)

        console.print(f"[bold]NFP Objective: {summary['id']}[/bold]")
        console.print(f"Title: {summary['title']}")
        console.print(f"Description: {summary['description']}")
        console.print(f"Commits: {summary['base_commit']} → {summary['target_commit']}")
        console.print(f"Created: {summary['created_at']}")
        console.print(f"Updated: {summary['updated_at']}")

        if nfp.user_stories:
            console.print(f"\\n[bold]User Stories ({len(nfp.user_stories)})[/bold]")

            for priority in range(1, 6):
                stories = nfp_service.get_stories_by_priority(nfp, priority)
                if stories:
                    priority_name = nfp_service.get_priority_name(priority)
                    console.print(f"\\n[bold]Priority {priority} ({priority_name})[/bold]")

                    for story in stories:
                        console.print(f"  • [cyan]{story.title}[/cyan]")
                        console.print(f"    {story.description}")
                        if story.acceptance_criteria:
                            console.print(f"    Acceptance: {', '.join(story.acceptance_criteria)}")
                        if story.tags:
                            console.print(f"    Tags: {', '.join(story.tags)}")
                        console.print(f"    Complexity: {story.estimated_complexity}")
                        console.print("")
        else:
            console.print("\\n[yellow]No user stories defined[/yellow]")

        console.print("\\n[bold]Summary[/bold]")
        console.print(f"Total Stories: {summary['story_count']}")
        console.print(f"Total Estimated Complexity: {summary['total_estimated_complexity']}")
        console.print(f"High Priority Stories: {summary['high_priority_stories_count']}")

    except Exception as e:
        console.print(f"[red]Failed to show NFP: {e}[/red]")


@summoner.command("delete-nfp")
@click.argument("nfp_id", shell_complete=complete_nfp_id)
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
def delete_nfp(nfp_id: str, yes: bool):
    """Delete an NFP objective and all its user stories."""
    nfp_service = NFPService()

    if not yes:
        nfp = nfp_service.get_nfp_objective(nfp_id)
        if not nfp:
            console.print(f"[red]NFP objective {nfp_id} not found[/red]")
            return

        console.print(f"[yellow]About to delete NFP: {nfp.title}[/yellow]")
        console.print(f"This will delete {nfp.story_count} user stories.")

        if not click.confirm("Are you sure?"):
            console.print("Cancelled")
            return

    try:
        if nfp_service.delete_nfp_objective(nfp_id):
            console.print(f"[green]✓ Deleted NFP objective {nfp_id}[/green]")
        else:
            console.print(f"[red]NFP objective {nfp_id} not found[/red]")

    except Exception as e:
        console.print(f"[red]Failed to delete NFP: {e}[/red]")
