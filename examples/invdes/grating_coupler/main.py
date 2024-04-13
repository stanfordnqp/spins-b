if __name__ == "__main__":
    import argparse
    from grating import run_opt, view_opt, view_opt_quick, resume_opt, gen_gds

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=("run", "view", "view_quick", "resume", "gen_gds"),
        help="Must be either \"run\" to run an optimization, \"view\" to "
             "view the results, \"resume\" to resume an optimization, or "
             "\"gen_gds\" to generate the grating GDS file.")
    parser.add_argument(
        "save_folder", help="Folder containing optimization logs.")

    grating_len = 12000
    wg_width = 12000

    args = parser.parse_args()
    if args.action == "run":
        run_opt(args.save_folder, grating_len=grating_len, wg_width=wg_width)
    elif args.action == "view":
        view_opt(args.save_folder)
    elif args.action == "view_quick":
        view_opt_quick(args.save_folder)
    elif args.action == "resume":
        resume_opt(args.save_folder)
    elif args.action == "gen_gds":
        gen_gds(args.save_folder, grating_len=grating_len, wg_width=wg_width)
