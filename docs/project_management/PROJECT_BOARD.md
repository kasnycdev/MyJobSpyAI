# Project Board Management

This document outlines how to manage the MyJobSpyAI project board, including workflows, issue tracking, and best practices.

## Board Structure

The project board is organized into the following columns:

1. **Backlog**
   - New issues that haven't been prioritized
   - Issues that are blocked or on hold
   - Future enhancements

2. **To Do**
   - Prioritized issues ready to be worked on
   - Issues with clear acceptance criteria
   - Dependencies are resolved

3. **In Progress**
   - Actively being worked on
   - Should have an assignee
   - Linked to a feature branch

4. **In Review**
   - Code changes submitted as PRs
   - Awaiting code review
   - May need additional changes

5. **Done**
   - Completed and merged
   - Ready for release
   - Documentation updated

## Workflow

### Creating New Issues
1. Use the appropriate issue template
2. Add relevant labels (e.g., `enhancement`, `bug`, `documentation`)
3. Add to the appropriate project board
4. Set priority using the `priority` label if needed

### Moving Between Columns
- **To Do** → **In Progress**: When work begins on the issue
- **In Progress** → **In Review**: When a PR is opened
- **In Review** → **Done**: When PR is approved and merged

### Best Practices
- Keep issues small and focused
- Update status regularly
- Link related issues and PRs
- Use milestones for tracking releases
- Add estimates if possible

## Automation

### GitHub Actions
Automated workflows handle:
- Moving issues to "In Review" when a PR is opened
- Moving to "Done" when PR is merged
- Adding appropriate labels based on PR content

### Label Conventions
- `bug`: For issues that report bugs
- `enhancement`: For new features or improvements
- `documentation`: For documentation updates
- `phase 1`, `phase 2`, `phase 3`: For project phases
- `priority: high`: For critical issues
- `wontfix`: For issues that won't be addressed

## Access
- [Project Board](https://github.com/users/kasnycdev/projects/3)
- [Issues](https://github.com/kasnycdev/MyJobSpyAI/issues)
- [Milestones](https://github.com/kasnycdev/MyJobSpyAI/milestones)
