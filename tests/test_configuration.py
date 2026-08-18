from configuration.configuration import ConfigurationManager


def test_configuration():

    configuration = ConfigurationManager()

    print("\nProject Name:")
    print(configuration.project_name)

    print("\nEnvironment:")
    print(configuration.environment)

    config = configuration.read_yaml()

    print("\nYAML Configuration:")
    print(config)


if __name__ == "__main__":
    test_configuration()