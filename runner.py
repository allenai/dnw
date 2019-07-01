# A file which enables command line arguments to be added dynamically when using a yaml file.
# Mostly, this is just a wrapper which runs train.main() with --gpus and --data_dir argument.
import os
import yaml
import sys


def parse_as_type(n):
    if type(n) is list:
        return [parse_as_type(x) for x in n]

    try:
        return int(n)
    except ValueError:
        pass

    try:
        return float(n)
    except ValueError:
        pass

    return n


def main():
    if len(sys.argv[1]) >= 4 and sys.argv[1][:4] == "app:":
        yaml_file = sys.argv[1].split(":")[-1]
        has_yaml_file = True
    else:
        yaml_file = "apps/default_cifar.yml"
        has_yaml_file = False

    with open(yaml_file, "r") as f:
        yaml_map = yaml.safe_load(f)

    if "/" in yaml_file:
        yaml_file = yaml_file.split("/")[-1]
    argmap = {"base": yaml_file.replace(".yml", "")}
    i = 2 if has_yaml_file else 1
    while i < len(sys.argv):
        k = sys.argv[i][2:]
        j = i + 1
        while (
            j < len(sys.argv) - 1
            and not sys.argv[j + 1][0] == "-"
            and not sys.argv[j][0] == "-"
        ):
            j = j + 1
        if j >= len(sys.argv) or sys.argv[j][0] == "-":
            if len(k) > 2 and k[:2] == "no":
                argmap[k[2:]] = False
            else:
                argmap[k] = True
        elif j == i + 1:
            argmap[k] = parse_as_type(sys.argv[j])
        else:
            argmap[k] = parse_as_type(sys.argv[i + 1 : j + 1])
        i = j if j < len(sys.argv) and sys.argv[j][0] == "-" else j + 1

    # Set environment variables.
    if "gpus" in argmap:
        if type(argmap["gpus"]) is not list:
            argmap["gpus"] = [argmap["gpus"]]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in argmap["gpus"])
        del argmap["gpus"]
    if "data-dir" in argmap:
        os.environ["DATA_DIR"] = argmap["data-dir"]
        del argmap["data-dir"]
    if "save_dir" in argmap:
        del argmap["save_dir"]
    if "log_dir" in argmap:
        del argmap["log_dir"]

    title = "_".join([k + "=" + str(v).replace(" ", "") for k, v in argmap.items()])
    argmap["title"] = title

    # write yaml.
    yaml_map.update(argmap)
    if not os.path.exists("apps/gen/"):
        os.mkdir("apps/gen/")
    new_yaml_file = os.path.join("apps/gen/", "{}.yml".format(title.replace("/", ".")))

    with open(new_yaml_file, "w") as f:
        yaml.dump(yaml_map, f)

    # run
    from genutil.config import reset_app

    reset_app(new_yaml_file)
    import train as train

    train.main()


if __name__ == "__main__":
    main()
