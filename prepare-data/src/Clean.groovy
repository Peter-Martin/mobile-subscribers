/**
 * Cleans CSV data file; i.e., fixes invalid values, etc.
 */

@Grab('com.xlson.groovycsv:groovycsv:1.1')

import static com.xlson.groovycsv.CsvParser.parseCsv
import com.xlson.groovycsv.PropertyMapper

class Clean {

    static void main(args) {

        // smaller sample for trying out changes/fixes
//        String inputCsvPath = '../50-original.csv'
//        File outputCsvPath = new File('../50.csv')

        String inputCsvPath = '../all-original.csv'
        File outputCsvPath = new File('../all.csv')

        outputCsvPath.delete() // clear out any existing contents

        // read header (first) line and output verbatim
        String headerLine
        new File(inputCsvPath).withReader { headerLine = it.readLine() }
        outputCsvPath << headerLine

        // read subscriber lines, fix values, then output
        for (PropertyMapper line in parseCsv(new FileReader(inputCsvPath), separator: ',')) {
            MobileSubscriber sub = line.toMap()
            outputCsvPath << '\n' << sub.fix().toCsvString()
        }
    }
}