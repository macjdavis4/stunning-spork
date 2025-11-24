"""
Single-stock trading demo for FINCON system.

Demonstrates trading a single stock with the multi-agent system.
"""

from fincon.config import FinconConfig
from fincon.training import create_trainer_from_config


def main():
    """Run single-stock demo."""

    # Create configuration
    config = FinconConfig(
        symbols=["AAPL"],
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
        enable_stock_selection_analyst=False
    )

    print("="*60)
    print("FINCON Single-Stock Demo")
    print("="*60)
    print(f"Symbol: {config.symbols[0]}")
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
