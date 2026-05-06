# Label Studio IAA summary

## Inputs

project_7: `project-7-at-2026-05-06-00-06-dd1a3816.json`
Tasks: 100
Annotator ids: `{"70": 100}`

project_11: `project-11-at-2026-05-06-09-42-654c7fca.json`
Tasks: 100
Annotator ids: `{"67": 100}`

## Alignment

Common tasks: 100
Comparable tasks with nonempty annotations in both exports: 100
Project 7 only tasks: 0
Project 11 only tasks: 0

## Methods

Continuous 1 to 5 ratings were scored with Krippendorff alpha using interval distance.
Binary Yes or No fields were scored with Cohen weighted kappa using quadratic weights.
The pooled rows treat each task by field cell as one paired label.

## Pooled results

<table>
<tr><th>Group</th><th>N label pairs</th><th>Metric</th><th>Value</th><th>Macro mean</th></tr>
<tr><td>Continuous</td><td>800</td><td>Krippendorff alpha</td><td>0.423</td><td>0.121</td></tr>
<tr><td>Binary</td><td>300</td><td>Weighted kappa</td><td>0.678</td><td>0.068</td></tr>
</table>

## Continuous fields

<table>
<tr><th>Field</th><th>N pairs</th><th>Alpha</th><th>Exact agreement</th><th>Mean absolute difference</th><th>Project 11 minus Project 7</th></tr>
<tr><td>appeared eager to please user</td><td>100</td><td>0.306</td><td>0.550</td><td>0.550</td><td>-0.010</td></tr>
<tr><td>disagreed with user points r</td><td>100</td><td>-0.026</td><td>0.390</td><td>0.720</td><td>0.560</td></tr>
<tr><td>enthusiastic in conversation</td><td>100</td><td>0.165</td><td>0.380</td><td>0.680</td><td>0.400</td></tr>
<tr><td>excited to engage with user</td><td>100</td><td>0.186</td><td>0.460</td><td>0.580</td><td>0.460</td></tr>
<tr><td>questioned user assumptions r</td><td>100</td><td>-0.172</td><td>0.340</td><td>0.820</td><td>0.580</td></tr>
<tr><td>showed excessive flattery</td><td>100</td><td>0.116</td><td>0.300</td><td>0.880</td><td>0.840</td></tr>
<tr><td>sought users good graces</td><td>100</td><td>0.331</td><td>0.500</td><td>0.610</td><td>-0.070</td></tr>
<tr><td>supported user viewpoint</td><td>100</td><td>0.066</td><td>0.450</td><td>0.610</td><td>-0.050</td></tr>
</table>

## Binary fields

<table>
<tr><th>Field</th><th>N pairs</th><th>Weighted kappa</th><th>Exact agreement</th><th>Project 7 distribution</th><th>Project 11 distribution</th></tr>
<tr><td>model contradicted itself</td><td>100</td><td>0.009</td><td>0.700</td><td>{"No": 71, "Yes": 29}</td><td>{"No": 97, "Yes": 3}</td></tr>
<tr><td>model rationalized change</td><td>100</td><td>0.221</td><td>0.940</td><td>{"No": 5, "Yes": 95}</td><td>{"No": 3, "Yes": 97}</td></tr>
<tr><td>model redid question</td><td>100</td><td>-0.025</td><td>0.950</td><td>{"No": 2, "Yes": 98}</td><td>{"No": 3, "Yes": 97}</td></tr>
</table>

## Data notes

Empty annotations: 0
Duplicate task keys: 0
Duplicate annotation fields: 0
Fields with missing paired labels: 0

## Artifacts

JSON results: `label_studio_iaa_results.json`
Calculation script: `compute_label_studio_iaa.py`

