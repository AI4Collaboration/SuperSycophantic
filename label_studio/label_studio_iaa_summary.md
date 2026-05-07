# Label Studio IAA summary

## Inputs

Reviewer A
Tasks: 100
Nonempty annotations: 100

Reviewer B
Tasks: 100
Nonempty annotations: 100

## Alignment

Common tasks: 100
Comparable tasks with nonempty annotations in both exports: 100
Reviewer A only tasks: 0
Reviewer B only tasks: 0

## Methods

Continuous 1 to 5 ratings were scored with Krippendorff alpha using interval distance.
Continuous fields also include Gwet AC2, ordinal quadratic weighted kappa, ICC(A,1), exact agreement, within one point agreement, mean absolute difference, and average signed difference.
Binary Yes or No fields were scored with Cohen weighted kappa using quadratic weights.
Binary fields also include Gwet AC1, PABAK, MCC, positive agreement, and negative agreement.
The pooled rows treat each task by field cell as one paired label.

## Pooled results

Continuous pooled metrics:

<table>
<tr><th>Metric</th><th>N label pairs</th><th>Pooled value</th><th>Macro mean</th></tr>
<tr><td>Krippendorff alpha</td><td>800</td><td>0.423</td><td>0.121</td></tr>
<tr><td>Gwet AC2</td><td>800</td><td>0.733</td><td>0.742</td></tr>
<tr><td>Ordinal weighted kappa</td><td>800</td><td>0.443</td><td>0.199</td></tr>
<tr><td>ICC(A,1)</td><td>800</td><td>0.443</td><td>0.200</td></tr>
<tr><td>Exact agreement</td><td>800</td><td>0.421</td><td>NA</td></tr>
<tr><td>Within one point agreement</td><td>800</td><td>0.907</td><td>0.907</td></tr>
<tr><td>Mean absolute difference</td><td>800</td><td>0.681</td><td>NA</td></tr>
</table>

Binary pooled metrics:

<table>
<tr><th>Metric</th><th>N label pairs</th><th>Pooled value</th><th>Macro mean</th></tr>
<tr><td>Weighted kappa</td><td>300</td><td>0.678</td><td>0.068</td></tr>
<tr><td>Gwet AC1</td><td>300</td><td>0.764</td><td>0.824</td></tr>
<tr><td>PABAK</td><td>300</td><td>0.727</td><td>0.727</td></tr>
<tr><td>MCC</td><td>300</td><td>0.692</td><td>0.073</td></tr>
<tr><td>Exact agreement</td><td>300</td><td>0.863</td><td>NA</td></tr>
<tr><td>Positive agreement</td><td>300</td><td>0.902</td><td>0.669</td></tr>
<tr><td>Negative agreement</td><td>300</td><td>0.773</td><td>0.357</td></tr>
</table>

## Interpretation

Coverage is complete: 100 common tasks and 100 paired nonempty annotations.
Continuous ratings show stronger coarse agreement than exact agreement: pooled Gwet AC2 is 0.733, within one point agreement is 0.907, and exact agreement is 0.421.
Field level continuous reliability remains uneven: macro mean Krippendorff alpha is 0.121, macro mean Gwet AC2 is 0.742, and macro mean ICC(A,1) is 0.200.
Binary labels have high raw agreement under class imbalance: exact agreement is 0.863, Gwet AC1 is 0.764, and PABAK is 0.727.
Minority class reliability is weaker: macro mean positive agreement is 0.669, macro mean negative agreement is 0.357, and macro mean MCC is 0.073.
Overall, the IAA supports complete coverage of the 100-item calibration sample and usable coarse binary audit labels, with weaker evidence for fine grained field level ordinal claims. It does not claim complete coverage of every shipped panel item.

## Continuous fields

<table>
<tr><th>Field</th><th>N pairs</th><th>Alpha</th><th>Gwet AC2</th><th>Ordinal weighted kappa</th><th>ICC(A,1)</th><th>Exact</th><th>Within one point</th><th>Mean absolute difference</th><th>Reviewer B minus Reviewer A</th></tr>
<tr><td>appeared eager to please user</td><td>100</td><td>0.306</td><td>0.737</td><td>0.302</td><td>0.304</td><td>0.550</td><td>0.910</td><td>0.550</td><td>-0.010</td></tr>
<tr><td>disagreed with user points r</td><td>100</td><td>-0.026</td><td>0.753</td><td>0.120</td><td>0.121</td><td>0.390</td><td>0.890</td><td>0.720</td><td>0.560</td></tr>
<tr><td>enthusiastic in conversation</td><td>100</td><td>0.165</td><td>0.798</td><td>0.225</td><td>0.227</td><td>0.380</td><td>0.940</td><td>0.680</td><td>0.400</td></tr>
<tr><td>excited to engage with user</td><td>100</td><td>0.186</td><td>0.829</td><td>0.277</td><td>0.279</td><td>0.460</td><td>0.960</td><td>0.580</td><td>0.460</td></tr>
<tr><td>questioned user assumptions r</td><td>100</td><td>-0.172</td><td>0.685</td><td>-0.013</td><td>-0.013</td><td>0.340</td><td>0.880</td><td>0.820</td><td>0.580</td></tr>
<tr><td>showed excessive flattery</td><td>100</td><td>0.116</td><td>0.667</td><td>0.286</td><td>0.288</td><td>0.300</td><td>0.840</td><td>0.880</td><td>0.840</td></tr>
<tr><td>sought users good graces</td><td>100</td><td>0.331</td><td>0.715</td><td>0.329</td><td>0.331</td><td>0.500</td><td>0.900</td><td>0.610</td><td>-0.070</td></tr>
<tr><td>supported user viewpoint</td><td>100</td><td>0.066</td><td>0.756</td><td>0.063</td><td>0.064</td><td>0.450</td><td>0.940</td><td>0.610</td><td>-0.050</td></tr>
</table>

## Binary fields

<table>
<tr><th>Field</th><th>N pairs</th><th>Weighted kappa</th><th>Gwet AC1</th><th>PABAK</th><th>MCC</th><th>Exact</th><th>Positive agreement</th><th>Negative agreement</th><th>Reviewer A distribution</th><th>Reviewer B distribution</th></tr>
<tr><td>model contradicted itself</td><td>100</td><td>0.009</td><td>0.590</td><td>0.400</td><td>0.017</td><td>0.700</td><td>0.062</td><td>0.821</td><td>{"No": 71, "Yes": 29}</td><td>{"No": 97, "Yes": 3}</td></tr>
<tr><td>model rationalized change</td><td>100</td><td>0.221</td><td>0.935</td><td>0.880</td><td>0.229</td><td>0.940</td><td>0.969</td><td>0.250</td><td>{"No": 5, "Yes": 95}</td><td>{"No": 3, "Yes": 97}</td></tr>
<tr><td>model redid question</td><td>100</td><td>-0.025</td><td>0.947</td><td>0.900</td><td>-0.025</td><td>0.950</td><td>0.974</td><td>0.000</td><td>{"No": 2, "Yes": 98}</td><td>{"No": 3, "Yes": 97}</td></tr>
</table>

## Data notes

Empty annotations: 0
Duplicate task keys: 0
Duplicate annotation fields: 0
Fields with missing paired labels: 0

## Artifacts

JSON results: `label_studio_iaa_results.json`
Calculation script: `compute_label_studio_iaa.py`

