# Job execution using bundles

We already have the job in a yaml file. To run the job directly in your terminal you need the following requirements:

- Databricks CLI - [documentation](https://docs.databricks.com/aws/en/dev-tools/cli/install)
- databricks CLI authenticated
- git folder in databricks with this project


Validate the bundle:
```bash
databricks bundle validate -t dev
```

Deploy the job to your workspace:
```bash
databricks bundle deploy -t dev
```

Execute the bundle-managed job from the CLI
```bash
databricks bundle run -t dev house_price_training
```

Delete the created infrastructure, in this case a job:
```bash
databricks bundle destroy -t dev 
```


