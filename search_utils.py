def print_results(results: list[dict], label: str = ""):
    if label:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")

    for r in results:
        print(f"  #{r['rank']} [score={r['score']:.4f}] {r['title']}")
        print(f"     {r['overview']}...")
        print()
