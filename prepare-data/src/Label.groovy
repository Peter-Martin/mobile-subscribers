/**
 * Applies labels to features in CSV file, according to various rules.
 */

@Grab('com.xlson.groovycsv:groovycsv:1.1')

import static com.xlson.groovycsv.CsvParser.parseCsv
import com.xlson.groovycsv.PropertyMapper

class Label {

    static String LABELS = ',UpdatedIn90Days,PurchasedAdditionalIn90Days'

    static void main(args) {

        // smaller sample for trying out changes/fixes
//        String inputCsvPath = '../50.csv'
//        File outputCsvPath = new File('../50-labelled.csv')

        String inputCsvPath = '../all.csv'
        File outputCsvPath = new File('../all-labelled.csv')

        outputCsvPath.delete() // clear out any existing contents

        // read header (first) line and output verbatim + labels
        String headerLine
        new File(inputCsvPath).withReader { headerLine = it.readLine() }
        outputCsvPath << headerLine << LABELS

        // read subscriber lines, fix values, then output
        for (PropertyMapper line in parseCsv(new FileReader(inputCsvPath), separator: ',')) {
            LabelledMobileSubscriber labelledSub = new LabelledMobileSubscriber(sub: line.toMap() as MobileSubscriber)
            outputCsvPath << '\n' << labelledSub.label().toCsvString()
        }
    }
}
