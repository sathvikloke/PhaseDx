# Task index

Generated from `tasks.json`. Task counts are not completion percentages or measured worker concurrency.

Execution is owned by the user and their authorized executor (currently Codex for CPU work). Scientific and author decisions stay with the user/Codex or a qualified human. No collaborator compute allocation.

## Execution tasks

| ID | Deliverable | Lane | Status | Dependencies |
|---|---|---|---|---|
| [E01](tasks/E01.md) | Bring the current study implementation into a portable repository layout | implementation | done | none |
| [E02](tasks/E02.md) | Locate the exact private inputs and checkpoint files | inputs | done | none |
| [E03](tasks/E03.md) | Create reproducible CPU and inference environments | implementation | done | none |
| [E04](tasks/E04.md) | Provide resumable execution with measured resource admission | execution | done | E01, E03 |
| [E05](tasks/E05.md) | Prepare an optional Einstein Slurm execution profile | HPC | optional | E01, E03 |
| [E06](tasks/E06.md) | Repair the three remaining acquisition edge cases once | acquisition | done | none |
| [E07](tasks/E07.md) | Prepare the executable image-only acquisition operation | acquisition | done | E01, E02, E06 |
| [E08](tasks/E08.md) | Acquire the CQ500 image archives | acquisition | done | E04, E07, H02 |
| [E09](tasks/E09.md) | Extract verified images without changing the source population | acquisition | done | E08 |
| [E10](tasks/E10.md) | Build the full DICOM roster and geometry inventory | cohort | doing | E01, E09 |
| [E11](tasks/E11.md) | Assemble a blinded clinical series-selection packet | cohort | doing | E10 |
| [E12](tasks/E12.md) | Apply the fixed clinical series selector | cohort | blocked | E01, E11, H03 |
| [E13](tasks/E13.md) | Authenticate the external reference and source-key schema | reference | blocked | none |
| [E14](tasks/E14.md) | Ingest the authorized reader reference after prediction freeze | reference | blocked | E13, E22, E23, E25, H04 |
| [E15](tasks/E15.md) | Bind person/examination grouping using legitimate source evidence | reference | blocked | E02, E13 |
| [E16](tasks/E16.md) | Bind the frozen CT model, comparator and executed preprocessing identities | model | done | E02 |
| [E17](tasks/E17.md) | Finish the portable frozen image inference CLI | model | done | E01, E03, E16 |
| [E18](tasks/E18.md) | Finish the transported frozen geometry inference CLI | model | done | E01, E16 |
| [E19](tasks/E19.md) | Make the paired scorer enforce the fixed study contract | statistics execution | done | E01 |
| [E20](tasks/E20.md) | Port deterministic missing-prediction completion bounds | statistics execution | done | E01 |
| [E21](tasks/E21.md) | Complete one end-to-end synthetic execution rehearsal | integration | done | E01, E04, E17, E18, E19, E20 |
| [E22](tasks/E22.md) | Freeze the clinical execution package before outcomes | integration | blocked | E12, E16, E17, E18, H01 |
| [E23](tasks/E23.md) | Run frozen seed42 external image predictions | clinical inference | blocked | E04, E08, E09, E12, E17, E21, E22, H02 |
| [E24](tasks/E24.md) | Run frozen seed123 descriptive external predictions | clinical inference | blocked | E04, E12, E17, E21, E22, H02 |
| [E25](tasks/E25.md) | Run the frozen transported geometry comparator | clinical inference | blocked | E04, E12, E18, E21, E22, H02 |
| [E26](tasks/E26.md) | Execute the prespecified external primary comparison | statistics execution | blocked | E14, E15, E19, E23, E25, H05 |
| [E27](tasks/E27.md) | Produce the prespecified descriptive secondary result | statistics execution | blocked | E14, E19, E24, E25 |
| [E28](tasks/E28.md) | Produce full-cohort flow, support and missing-score results | cohort results | blocked | E10, E12, E14, E20, E23, E25 |
| [E29](tasks/E29.md) | Generate external-study tables and figure data mechanically | presentation execution | blocked | E26, E27, E28 |
| [E30](tasks/E30.md) | Recover source-bound MRI training and preprocessing Methods | historical Methods | done | none |
| [E31](tasks/E31.md) | Recover breast target, case mapping and executed Methods | historical Methods | done | none |
| [E32](tasks/E32.md) | Recover internal CT model, selection and preprocessing Methods | historical Methods | done | none |
| [E33](tasks/E33.md) | Complete retained-cohort selection and exposure chronology | historical Methods | done | E02 |
| [E34](tasks/E34.md) | Assemble the accepted aggregate result inputs for the revised paper | paper inputs | done | E02 |
| [E35](tasks/E35.md) | Package the completed synthetic studies for reproducible use | paper inputs | done | E02 |
| [E36](tasks/E36.md) | Reconcile the older public study with the current submission scope | historical inventory | done | none |
| [E37](tasks/E37.md) | Provide a clean-checkout reproducibility smoke path | release | done | E01, E03, E17, E18, E19, E20, E35 |
| [E38](tasks/E38.md) | Render approved figures and tables at submission quality | presentation execution | blocked | E29, E34, E35, H10 |
| [E39](tasks/E39.md) | Collect verifiable reference metadata and Methods source excerpts | paper inputs | done | none |
| [E40](tasks/E40.md) | Assemble the reporting and disclosure evidence matrix | paper inputs | done | E30, E31, E32, E33 |
| [E41](tasks/E41.md) | Populate the approved author and disclosure metadata mechanically | submission assembly | blocked | H12, H13 |
| [E42](tasks/E42.md) | Build the final journal files from scientifically approved text | submission assembly | blocked | E38, E40, E41, H10, H11 |
| [E43](tasks/E43.md) | Prepare the reproducible public code and aggregate release | release | doing | E34, E35, E37, H12 |
| [E44](tasks/E44.md) | Prepare a checkpoint availability package if redistribution is permitted | release | blocked | E16, H12 |
| [E45](tasks/E45.md) | Return a compact evidence bundle for independent scientific review | review handoff | doing | E26, E27, E28, E30, E31, E32, E33, E34, E35 |
| [E46](tasks/E46.md) | Execute only a specifically adjudicated result-changing correction | conditional repair | optional | H07 |
| [E47](tasks/E47.md) | Recover a retained historical benchmark artifact only if absent | conditional historical | optional | E36, H01 |
| [E48](tasks/E48.md) | Recover permissible full texts if the historical prevalence study is retained | optional screen | optional | E36, H01 |
| [E49](tasks/E49.md) | Recompute the historical screen outputs after independent human coding | optional screen | optional | E48, H15 |
| [E50](tasks/E50.md) | Publish only the explicitly approved release artifacts | release | blocked | E43, E44, H14 |

## Scientific and author tasks

| ID | Deliverable | Lane | Status | Dependencies |
|---|---|---|---|---|
| [H01](tasks/H01.md) | Confirm the current paper scope and uphold the frozen external design | scientific / author | done | none |
| [H02](tasks/H02.md) | Resolve the concrete current acquisition/inference scope and cumulative baseline | scientific / author | done | E07 |
| [H03](tasks/H03.md) | Resolve ambiguous clinical series compatibility while blinded | scientific / author | decision_needed | E11 |
| [H04](tasks/H04.md) | Approve the exact external reference/source-key interpretation | scientific / author | decision_needed | E13 |
| [H05](tasks/H05.md) | Adjudicate the supported resampling unit | scientific / author | decision_needed | E15 |
| [H06](tasks/H06.md) | Resolve clinical meaning and reference limits of retained MRI/breast/CT targets | scientific / author | decision_needed | E30, E31, E32, E33 |
| [H07](tasks/H07.md) | Independently verify the new external numerical result | scientific / author | decision_needed | E26, E27, E28 |
| [H08](tasks/H08.md) | Adjudicate the completed synthetic findings for the current paper | scientific / author | done | E35 |
| [H09](tasks/H09.md) | Interpret external evidence against the stated question | scientific / author | decision_needed | H07, H05 |
| [H10](tasks/H10.md) | Write and validate the revised scientific manuscript | scientific / author | doing | H06, H08, H09, E34 |
| [H11](tasks/H11.md) | Assess current novelty and journal fit from primary sources | scientific / author | done | E39 |
| [H12](tasks/H12.md) | Confirm ethics, permitted use and release statements | scientific / author | decision_needed | none |
| [H13](tasks/H13.md) | Supply the final author roster and approvals | scientific / author | decision_needed | none |
| [H14](tasks/H14.md) | Perform final scientific and author submission review | scientific / author | decision_needed | H10, H11, H12, H13, E42, E43 |
| [H15](tasks/H15.md) | Provide genuinely independent coding if the older screen is included | scientific / author | optional | E48 |
