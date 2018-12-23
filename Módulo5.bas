Attribute VB_Name = "Módulo5"
Sub Create_Pivot_Table_COE()
'Todas las lineas de solicitudes que:
'COE
'Pais Chile

Mensaje = MsgBox("Creara las tablas pivot" & vbCrLf & "Mostrando Sourcing y Chile", vbOKCancel)
If Mensaje = vbCancel Then
Exit Sub
End If

'https://www.thespreadsheetguru.com/blog/2014/9/27/vba-guide-excel-pivot-tables
'http://www.globaliconnect.com/excel/index.php?option=com_content&view=article&id=150:excel-pivot-tables-filter-data-items-values-a-dates-using-vba&catid=79&Itemid=475

'Declare Variables
Dim PSheet As Worksheet 'To create a sheet for a new pivot table.
Dim DSheet As Worksheet 'To use as a data sheet.
Dim PCache As PivotCache 'To use as a name for pivot table cache.
Dim PTable As PivotTable 'To use as a name for our pivot table.
Dim PRange As Range 'to define source data range.
Dim lastRow As Long 'To get the last row and column of our data range.
Dim LastCol As Long 'To get the last row and column of our data range.

'Insert a New Blank Worksheet
On Error Resume Next
Application.DisplayAlerts = False
Worksheets("COE").Delete
Sheets.Add Before:=ActiveSheet
ActiveSheet.Name = "COE"
Application.DisplayAlerts = True
Set PSheet = Worksheets("COE")
Set DSheet = Worksheets("Base")

'Define Data Range
lastRow = DSheet.Cells(Rows.Count, 1).End(xlUp).Row
LastCol = DSheet.Cells(5, Columns.Count).End(xlToLeft).Column
Set PRange = DSheet.Cells(5, 1).Resize(lastRow, LastCol)

'Define Pivot Cache
Set PCache = ActiveWorkbook.PivotCaches.Create _
(SourceType:=xlDatabase, SourceData:=PRange). _
CreatePivotTable(TableDestination:=PSheet.Cells(2, 2), _
TableName:="COE")

'Insert Blank Pivot Table
Set PTable = PCache.CreatePivotTable _
(TableDestination:=PSheet.Cells(1, 1), TableName:="COE")

'Insert Report Filter (estos son los filtros)

With ActiveSheet.PivotTables("COE").PivotFields("Cantidad de lineas")
.Orientation = xlPageField
.PivotItems("1").Visible = True
        For i = 2 To .PivotItems.Count
            With .PivotItems(i)
                If i > 1 Then
                    .Visible = False
                End If
            End With
        Next i
.Position = 4
End With

With ActiveSheet.PivotTables("COE").PivotFields("Area de compra")
.Orientation = xlPageField
.PivotItems("COE").Visible = True
.PivotItems("Compra").Visible = False
.PivotItems("Gestion PO").Visible = False
.PivotItems("(blank)").Visible = False
.Position = 3
End With

With ActiveSheet.PivotTables("COE").PivotFields("Pais")
.Orientation = xlPageField
.PivotItems("Chile").Visible = True
.PivotItems("Perú").Visible = True
.PivotItems("(blank)").Visible = False
.Position = 2
End With

With ActiveSheet.PivotTables("COE").PivotFields("Tipo de compra")
.Orientation = xlPageField
.PivotItems("Catalogo").Visible = False
.PivotItems("Contrato").Visible = False
.PivotItems("Sourcing").Visible = True
.PivotItems("Politica").Visible = False
.PivotItems("(blank)").Visible = False
.Position = 1
End With

'Insert Column Fields  (estos son las columnas)
With ActiveSheet.PivotTables("COE").PivotFields("Dias Pen")
.Orientation = xlColumnField
.Position = 1
.PivotItems("").Visible = False
.Subtotals(1) = False
End With

'Insert Row Labels (estos son las filas)
With ActiveSheet.PivotTables("COE").PivotFields("Taxonomia")
.Orientation = xlRowField
.PivotItems("(blank)").Visible = False
.PivotItems("Hernan Guzman").Visible = True
.PivotItems("Cesar Yepez").Visible = True
.PivotItems("Maritza Vidal").Visible = True
.PivotItems("Maria Ramirez").Visible = True
.PivotItems("Elida flores").Visible = True
.PivotItems("Herman Guzmán").Visible = True
.PivotItems("Alejandra Barrera").Visible = False
.PivotItems("Analuz Arcaya").Visible = False
.PivotItems("Andrea Cofre").Visible = False
.PivotItems("Barbara Calderon").Visible = False
.PivotItems("Barbara Calderón").Visible = False
.PivotItems("Belen Carreño").Visible = False
.PivotItems("Berenise Balbontin").Visible = False
.PivotItems("Berenice Balbontin").Visible = False
.PivotItems("Camilo Santana").Visible = False
.PivotItems("Cinthia Gonzalez").Visible = False
.PivotItems("Cristian Farias").Visible = False
.PivotItems("Daniel Alvarez").Visible = False
.PivotItems("Daniel Becerra").Visible = False
.PivotItems("Daniel Inostroza").Visible = False
.PivotItems("Daniel Mardones").Visible = False
.PivotItems("Deborah Rozas").Visible = False
.PivotItems("Denisse Henriquez").Visible = False
.PivotItems("Ediver Flores").Visible = False
.PivotItems("Eiker Briceño").Visible = False
.PivotItems("Elizabeth Mancilla").Visible = False
.PivotItems("Erik Guzman").Visible = False
.PivotItems("Fernanda Gonzalez").Visible = False
.PivotItems("Fernando Gonzalez").Visible = False
.PivotItems("Karina Lepique").Visible = False
.PivotItems("Marcos Rodriguez").Visible = False
.PivotItems("Maria Latouche").Visible = False
.PivotItems("Matias Hinojosa").Visible = False
.PivotItems("Mauricio Valenzuela").Visible = False
.PivotItems("Nicole Fuentes").Visible = False
.PivotItems("Yoluimar Iznaga").Visible = False
.PivotItems("Solicitud Procesada").Visible = False
.Position = 1
.Subtotals(1) = False
End With

'Insert Data Field (estos son las valores)
With ActiveSheet.PivotTables("COE").PivotFields("Lineadistribucion")
.Orientation = xlDataField
.Position = 1
.Function = xlCount
.NumberFormat = "#,##0"
End With

'http://www.globaliconnect.com/excel/index.php?option=com_content&view=article&id=150:excel-pivot-tables-filter-data-items-values-a-dates-using-vba&catid=79&Itemid=475

'Format Pivot Table
'ActiveSheet.PivotTables("SalesPivotTable").ShowTableStyleRowStripes = True
'ActiveSheet.PivotTables("SalesPivotTable").TableStyle2 = "PivotStyleMedium9"

'Worksheets("COE").Range("A1:H22").Interior.ColorIndex = 2


End Sub



