"""
Portfolio trading demo for FINCON system.

Demonstrates portfolio optimization with the multi-agent system.
"""

from fincon.config import FinconConfig
from fincon.training import create_trainer_from_config


def main():
    """Run portfolio demo."""

    # Create configuration
    config = FinconConfig(
        symbols=["AAPL", "MSFT", "GOOGL"],
        start_date="2023-01-01",
        end_date="2023-06-30",
        episodes=3,
        initial_capital=100000.0,
        cvar_alpha=0.05,
        verbose=False,
        enable_news_analyst=True,
        enable_fundamentals_analyst=True,
        enable_data_analyst=True,
        enable_ecc_analyst=False,
        enable_stock_selection_analyst=True
    )

    print("="*60)
    print("FINCON Portfolio Demo")
    print("="*60)
    print(f"Symbols: {', '.join(config.symbols)}")
    print(f"Period: {config.start_date} to {config.end_date}")
    print(f"Episodes: {config.episodes}")
    print(f"Initial Capital: ${config.initial_capital:,.0f}")
    print("="*60)

    # Create and run trainer
    try:
        trainer = create_trainer_from_config(config)
        trainer.train()
        trainer.print_summary()

        print("\n" + "="*60)
        print("Demo complete!")
        print("="*60)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
