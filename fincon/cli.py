"""
Command-line interface for FINCON system.

Provides commands for:
- Single-stock trading
- Portfolio trading
- Config generation
"""

import argparse
import sys
from pathlib import Path

from fincon.config import FinconConfig, load_config, create_default_config
from fincon.training import create_trainer_from_config


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="FINCON: Multi-Agent Financial Decision System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single-stock trading
  python -m fincon single-stock AAPL --config config.json

  # Portfolio trading
  python -m fincon portfolio AAPL MSFT TSLA --config config.json

  # Generate default config
  python -m fincon create-config --output config.json
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Single-stock command
    single_parser = subparsers.add_parser(
        'single-stock',
        help='Run single-stock trading experiment'
    )
    single_parser.add_argument(
        'symbol',
        type=str,
        help='Stock ticker symbol'
    )
    single_parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration file'
    )
    single_parser.add_argument(
        '--start-date',
        type=str,
        help='Start date (YYYY-MM-DD), overrides config'
    )
    single_parser.add_argument(
        '--end-date',
        type=str,
        help='End date (YYYY-MM-DD), overrides config'
    )
    single_parser.add_argument(
        '--episodes',
        type=int,
        help='Number of episodes, overrides config'
    )
    single_parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )

    # Portfolio command
    portfolio_parser = subparsers.add_parser(
        'portfolio',
        help='Run portfolio trading experiment'
    )
    portfolio_parser.add_argument(
        'symbols',
        type=str,
        nargs='+',
        help='Stock ticker symbols'
    )
    portfolio_parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration file'
    )
    portfolio_parser.add_argument(
        '--start-date',
        type=str,
        help='Start date (YYYY-MM-DD), overrides config'
    )
    portfolio_parser.add_argument(
        '--end-date',
        type=str,
        help='End date (YYYY-MM-DD), overrides config'
    )
    portfolio_parser.add_argument(
        '--episodes',
        type=int,
        help='Number of episodes, overrides config'
    )
    portfolio_parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )

    # Create-config command
    config_parser = subparsers.add_parser(
        'create-config',
        help='Create default configuration file'
    )
    config_parser.add_argument(
        '--output',
        type=str,
        default='config.json',
        help='Output file path (default: config.json)'
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Handle commands
    if args.command == 'create-config':
        handle_create_config(args)
    elif args.command == 'single-stock':
        handle_single_stock(args)
    elif args.command == 'portfolio':
        handle_portfolio(args)
    else:
        parser.print_help()
        sys.exit(1)


def handle_create_config(args):
    """Handle create-config command."""
    output_path = args.output

    if Path(output_path).exists():
        response = input(f"{output_path} already exists. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("Aborted.")
            sys.exit(0)

    create_default_config(output_path)
    print(f"Created default configuration at: {output_path}")
    print("\nPlease edit the configuration file and set:")
    print("  - EDGAR user agent (edgar.user_agent)")
    print("  - Finlight API key (finlight.api_key)")
    print("  - LLM API key (llm.api_key)")
    print("  - LLM model name (llm.model)")


def handle_single_stock(args):
    """Handle single-stock command."""
    # Load config
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Error: Config file not found: {args.config}")
        print("Create one with: python -m fincon create-config")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

    # Override with CLI arguments
    config.symbols = [args.symbol]

    if args.start_date:
        config.start_date = args.start_date
    if args.end_date:
        config.end_date = args.end_date
    if args.episodes:
        config.episodes = args.episodes
    if args.verbose:
        config.verbose = True

    # Disable stock selection analyst for single stock
    config.enable_stock_selection_analyst = False

    # Run training
    try:
        trainer = create_trainer_from_config(config)
        trainer.train()
        trainer.print_summary()

    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def handle_portfolio(args):
    """Handle portfolio command."""
    # Load config
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Error: Config file not found: {args.config}")
        print("Create one with: python -m fincon create-config")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

    # Override with CLI arguments
    config.symbols = args.symbols

    if args.start_date:
        config.start_date = args.start_date
    if args.end_date:
        config.end_date = args.end_date
    if args.episodes:
        config.episodes = args.episodes
    if args.verbose:
        config.verbose = True

    # Enable stock selection analyst for portfolio
    config.enable_stock_selection_analyst = True

    # Run training
    try:
        trainer = create_trainer_from_config(config)
        trainer.train()
        trainer.print_summary()

    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
