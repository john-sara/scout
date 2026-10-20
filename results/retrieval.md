Retrieval on 30 answerable questions over 148 pages.

| Chunking | Chunks | Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---|---|---|---|---|
| fixed | 695 | 57% | 77% | 87% | 0.69 |
| headings | 1142 | 63% | 80% | 93% | 0.75 |

### Questions not found in the top 3 pages (fixed)

- q03: How do I make a query parameter optional?  expected ['tutorial/query-params.md'], got ['tutorial/dependencies/index.md', 'tutorial/query-params-str-validations.md', 'tutorial/body.md']
- q04: How do I limit how long a query string parameter can be?  expected ['tutorial/query-params-str-validations.md'], got ['tutorial/query-params.md', 'tutorial/query-param-models.md', 'tutorial/dependencies/classes-as-dependencies.md']
- q14: How do I write automated tests for my endpoints?  expected ['tutorial/testing.md'], got ['advanced/opentelemetry.md', 'advanced/events.md', 'advanced/async-tests.md']
- q16: How do I serve a folder of static files like CSS and images?  expected ['tutorial/static-files.md'], got ['reference/staticfiles.md', 'how-to/custom-docs-ui-assets.md', 'advanced/templates.md']
- q18: How do I attach a cookie to my response?  expected ['advanced/response-cookies.md'], got ['reference/responses.md', 'tutorial/cookie-param-models.md', 'tutorial/cookie-params.md']
- q24: How do I turn off the interactive docs for production?  expected ['how-to/conditional-openapi.md'], got ['tutorial/security/first-steps.md', 'reference/index.md', 'tutorial/first-steps.md']
- q30: How do I organize my endpoints into groups in the interactive docs?  expected ['tutorial/metadata.md', 'tutorial/path-operation-configuration.md'], got ['advanced/opentelemetry.md', 'features.md', 'deployment/docker.md']

### Questions not found in the top 3 pages (headings)

- q06: How do I make an endpoint respond with 201 instead of 200?  expected ['tutorial/response-status-code.md'], got ['advanced/response-change-status-code.md', 'tutorial/cors.md', 'advanced/websockets.md']
- q14: How do I write automated tests for my endpoints?  expected ['tutorial/testing.md'], got ['advanced/opentelemetry.md', 'advanced/async-tests.md', 'advanced/events.md']
- q16: How do I serve a folder of static files like CSS and images?  expected ['tutorial/static-files.md'], got ['reference/staticfiles.md', 'how-to/custom-docs-ui-assets.md', 'advanced/templates.md']
- q18: How do I attach a cookie to my response?  expected ['advanced/response-cookies.md'], got ['reference/responses.md', 'tutorial/cookie-param-models.md', 'tutorial/cookie-params.md']
- q24: How do I turn off the interactive docs for production?  expected ['how-to/conditional-openapi.md'], got ['index.md', 'tutorial/first-steps.md', 'deployment/docker.md']
- q30: How do I organize my endpoints into groups in the interactive docs?  expected ['tutorial/metadata.md', 'tutorial/path-operation-configuration.md'], got ['advanced/opentelemetry.md', 'index.md', 'tutorial/first-steps.md']
